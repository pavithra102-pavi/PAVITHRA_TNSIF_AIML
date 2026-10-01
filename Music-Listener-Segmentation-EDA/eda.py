import argparse
import json
import os
import re
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = "viridis"

NULL_TOKENS = {"", "nan", "na", "n/a", "null", "none", "?", "-", "--"}


# =============================================================== helpers
def snake(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower()


def md_table(df: pd.DataFrame, index=True) -> str:
    """Tiny markdown-table writer (no extra dependency)."""
    d = df.reset_index() if index else df
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(d.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.values]
    return "\n".join([head, sep] + body)


class Saver:
    """Saves numbered charts into the charts folder."""

    def __init__(self, folder):
        self.folder, self.count = folder, 0
        os.makedirs(folder, exist_ok=True)

    def save(self, fig, name):
        self.count += 1
        path = os.path.join(self.folder, f"{self.count:02d}_{name}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"   chart saved: {os.path.basename(path)}")


def grid(n, ncols=4, cell=(4, 3.2)):
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    return fig, axes


# =============================================================== 1. load + inspect
def load(path):
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"Could not read {path}")


def inspect(raw: pd.DataFrame) -> str:
    buf = [f"Rows: {raw.shape[0]}   Columns: {raw.shape[1]}", "", "COLUMN TYPES / MISSING / UNIQUE"]
    info = pd.DataFrame(
        {
            "dtype": raw.dtypes.astype(str),
            "missing": raw.isna().sum(),
            "missing_%": (raw.isna().mean() * 100).round(2),
            "unique": raw.nunique(),
        }
    )
    buf += [info.to_string(), "", f"Fully duplicated rows: {raw.duplicated().sum()}", "", "FIRST 5 ROWS", raw.head().to_string(), ""]
    buf += ["NUMERIC SUMMARY", raw.describe().T.round(2).to_string()]
    return "\n".join(buf)


# =============================================================== 2. clean + validate
def is_id_col(name, s, n):
    by_name = re.search(r"(^|_)(id|uuid)($|_)|^user$|^userid$", name) is not None
    unique_ratio = s.nunique(dropna=True) / max(n, 1)
    return (by_name and unique_ratio > 0.9) or (s.dtype == object and unique_ratio > 0.9)


def clean(raw: pd.DataFrame):
    """Returns cleaned dataframe + a dict describing every fix that was applied."""
    rep = {"steps": [], "counts": {}}
    df = raw.copy()
    n0 = len(df)

    # column names
    df.columns = [snake(c) for c in df.columns]
    rep["steps"].append("Column names converted to snake_case")

    # text cleanup
    obj_cols = df.select_dtypes(include="object").columns
    n_null_tokens = 0
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()
        mask = df[c].str.lower().isin(NULL_TOKENS)
        n_null_tokens += int(mask.sum())
        df.loc[mask, c] = pd.NA
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    rep["counts"]["placeholder_text_to_missing"] = n_null_tokens

    # numbers stored as text -> numeric
    converted = []
    for c in df.select_dtypes(include="object").columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        nn = df[c].notna().sum()
        if nn and conv.notna().sum() / nn >= 0.9:
            df[c] = conv
            converted.append(c)
    rep["counts"]["text_columns_converted_to_numeric"] = converted

    # duplicates
    dup_rows = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    rep["counts"]["duplicate_rows_removed"] = dup_rows
    id_cols = [c for c in df.columns if is_id_col(c, df[c], len(df))]
    dup_ids = 0
    for c in id_cols:
        if re.search(r"id", c):
            before = len(df)
            df = df.drop_duplicates(subset=c).reset_index(drop=True)
            dup_ids += before - len(df)
    rep["counts"]["duplicate_ids_removed"] = dup_ids

    # inconsistent category spelling (e.g. 'premium ', 'MOBILE')
    fixed_labels = 0
    for c in df.select_dtypes(include="object").columns:
        if c in id_cols:
            continue
        key = df[c].astype("string").str.lower().str.strip()
        canon = df.assign(_k=key).groupby("_k")[c].agg(lambda s: s.value_counts().index[0])
        new = key.map(canon)
        fixed_labels += int(((new != df[c]) & df[c].notna()).sum())
        df[c] = new.where(df[c].notna(), np.nan).astype(object)
    rep["counts"]["category_labels_standardised"] = fixed_labels

    # validation of impossible values
    num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in id_cols]
    invalid = {}
    for c in num_cols:
        s = df[c]
        bad = pd.Series(False, index=df.index)
        if (s.quantile(0.01) >= 0):  # column is naturally non-negative
            bad |= s < 0
        if s.quantile(0.99) <= 1.0 and re.search(r"rate|ratio|pct|percent|share", c):
            bad |= (s < 0) | (s > 1)
        if "daily" in c and re.search(r"min", c):
            bad |= s > 1440          # more minutes than exist in a day
        if "daily" in c and re.search(r"hour|hrs", c):
            bad |= s > 24
        if c in {"age", "user_age", "listener_age"}:
            bad |= (s < 10) | (s > 100)
        if bad.any():
            invalid[c] = int(bad.sum())
            df.loc[bad, c] = np.nan
    rep["counts"]["invalid_values_set_to_missing"] = invalid

    # missing values
    high_missing = [c for c in df.columns if df[c].isna().mean() > 0.5]
    df = df.drop(columns=high_missing)
    rep["counts"]["columns_dropped_over_50pct_missing"] = high_missing
    filled = {}
    for c in df.columns:
        m = int(df[c].isna().sum())
        if m == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(df[c].median())
            filled[c] = f"{m} -> median"
        else:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
            filled[c] = f"{m} -> mode"
    rep["counts"]["missing_values_filled"] = filled

    rep["counts"]["rows_before"], rep["counts"]["rows_after"] = n0, len(df)
    return df, rep


def split_columns(df):
    id_cols = [c for c in df.columns if is_id_col(c, df[c], len(df))]
    num = [c for c in df.select_dtypes(include=np.number).columns if c not in id_cols]
    cat, high_card = [], []
    for c in df.columns:
        if c in id_cols or c in num:
            continue
        (cat if df[c].nunique() <= 30 else high_card).append(c)
    return id_cols, num, cat, high_card


def outlier_table(df, num):
    rows = []
    for c in num:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((df[c] < lo) | (df[c] > hi)).sum())
        rows.append({"feature": c, "lower_fence": round(lo, 2), "upper_fence": round(hi, 2),
                     "outliers": n_out, "outlier_%": round(100 * n_out / len(df), 2)})
    return pd.DataFrame(rows).set_index("feature")


# =============================================================== 3. EDA charts
def eda_charts(raw, df, num, cat, saver):
    # missing values (before cleaning)
    miss = (raw.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if len(miss):
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(miss) + 1)))
        sns.barplot(x=miss.values, y=miss.index, palette=PALETTE, ax=ax)
        ax.set_xlabel("% missing (raw data)")
        ax.set_title("Missing values before cleaning")
        saver.save(fig, "missing_values")

    if num:
        fig, axes = grid(len(num))
        for ax, c in zip(axes, num):
            sns.histplot(df[c], kde=True, ax=ax, color="#3b7dd8")
            ax.axvline(df[c].mean(), color="red", ls="--", lw=1)
            ax.set_title(f"{c}\nskew={df[c].skew():.2f}", fontsize=10)
            ax.set_xlabel("")
        fig.suptitle("Distributions of listening-behaviour features (red line = mean)", y=1.01, fontsize=14)
        saver.save(fig, "distributions")

        fig, axes = grid(len(num))
        for ax, c in zip(axes, num):
            sns.boxplot(y=df[c], ax=ax, color="#8fc1e3", fliersize=3)
            ax.set_title(c, fontsize=10)
            ax.set_ylabel("")
        fig.suptitle("Boxplots - outlier check", y=1.01, fontsize=14)
        saver.save(fig, "boxplots_outliers")

    if cat:
        fig, axes = grid(len(cat), ncols=3, cell=(5, 3.4))
        for ax, c in zip(axes, cat):
            order = df[c].value_counts().index
            sns.countplot(y=df[c], order=order, palette=PALETTE, ax=ax)
            ax.set_title(c, fontsize=11)
            ax.set_ylabel("")
        fig.suptitle("Categorical features - listener counts", y=1.01, fontsize=14)
        saver.save(fig, "categorical_counts")

    if len(num) >= 2:
        corr = df[num].corr()
        size = max(6, 0.7 * len(num))
        fig, ax = plt.subplots(figsize=(size + 1, size))
        sns.heatmap(corr, annot=len(num) <= 14, fmt=".2f", cmap="coolwarm", center=0, square=True,
                    linewidths=0.5, ax=ax, annot_kws={"size": 8})
        ax.set_title("Correlation heatmap")
        saver.save(fig, "correlation_heatmap")

        pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(10)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = [f"{a} vs {b}" for a, b in top.index]
        sns.barplot(x=top.values, y=labels, palette="coolwarm", ax=ax)
        ax.set_title("Top 10 feature pairs by correlation")
        ax.set_xlabel("Pearson r")
        saver.save(fig, "top_correlations")

    # behaviour profile of each category (mean z-score of each numeric feature)
    if num:
        z = (df[num] - df[num].mean()) / df[num].std().replace(0, 1)
        used = 0
        for c in cat:
            if used >= 4 or not (2 <= df[c].nunique() <= 12):
                continue
            prof = z.groupby(df[c]).mean()
            fig, ax = plt.subplots(figsize=(max(7, 0.85 * len(num)), max(3, 0.55 * len(prof) + 1.5)))
            sns.heatmap(prof, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5, ax=ax,
                        annot_kws={"size": 8})
            ax.set_title(f"Listening behaviour by '{c}'  (average z-score; red = above overall average)")
            ax.set_ylabel("")
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
            saver.save(fig, f"behaviour_by_{c}")
            used += 1

    # pairwise scatter of the 4 most varied features
    if len(num) >= 3:
        cv = (df[num].std() / df[num].mean().abs().replace(0, np.nan)).sort_values(ascending=False)
        pick = list(cv.head(4).index)
        g = sns.pairplot(df[pick].sample(min(1000, len(df)), random_state=1), corner=True,
                         plot_kws={"alpha": 0.4, "s": 12}, diag_kind="kde", height=2.2)
        g.fig.suptitle(f"Pairwise relationships ({len(pick)} most variable features, {min(1000, len(df))} listeners)", y=1.02)
        saver.save(g.fig, "pairwise_scatter")


# =============================================================== 4. clustering prep
def prepare_for_clustering(df, id_cols, num, cat, out_dir):
    info = {"capped": [], "log_transformed": [], "one_hot": cat, "dropped": []}
    X = df[num].copy()

    for c in num:  # cap outliers at IQR fences so KMeans is not dragged by extremes
        q1, q3 = X[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        if ((X[c] < lo) | (X[c] > hi)).any():
            X[c] = X[c].clip(lo, hi)
            info["capped"].append(c)
    for c in num:  # reduce heavy right skew
        if X[c].min() >= 0 and X[c].skew() > 1.0:
            X[c] = np.log1p(X[c])
            info["log_transformed"].append(c)

    const = [c for c in X.columns if X[c].std() == 0]
    X = X.drop(columns=const)
    info["dropped"] += const

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    if cat:
        dummies = pd.get_dummies(df[cat], dtype=int, prefix_sep="=")
        scaled = pd.concat([scaled, dummies], axis=1)

    ready = scaled.copy()
    if id_cols:
        ready.insert(0, id_cols[0], df[id_cols[0]].values)
    ready.to_csv(os.path.join(out_dir, "music_listeners_clustering_ready.csv"), index=False)
    joblib.dump(scaler, os.path.join(out_dir, "scaler_eda_reference.joblib"))
    info["features"] = list(scaled.columns)
    with open(os.path.join(out_dir, "feature_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    return scaled, info


def clustering_preview(features, saver, num_features):
    """PCA + elbow + silhouette - tells the modelling teammate a sensible k range."""
    res = {}
    pca = PCA().fit(features)
    cum = np.cumsum(pca.explained_variance_ratio_)
    res["components_for_80pct"] = int(np.argmax(cum >= 0.80) + 1)
    res["pc1_pc2_variance"] = round(float(cum[min(1, len(cum) - 1)]) * 100, 1)

    sample = features.sample(min(3000, len(features)), random_state=1)
    ks = list(range(2, 11))
    inertia, sil, sil_num = [], [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(sample)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(sample, km.labels_))
        if num_features and len(num_features) < features.shape[1]:  # numeric-only run (only when one-hot columns exist)
            kn = KMeans(n_clusters=k, n_init=10, random_state=42).fit(sample[num_features])
            sil_num.append(silhouette_score(sample[num_features], kn.labels_))
    res["best_k_by_silhouette"] = ks[int(np.argmax(sil))]
    res["best_silhouette"] = round(float(max(sil)), 3)
    res["silhouettes"] = {k: round(float(s), 3) for k, s in zip(ks, sil)}
    if sil_num:
        res["best_k_numeric_only"] = ks[int(np.argmax(sil_num))]
        res["best_silhouette_numeric_only"] = round(float(max(sil_num)), 3)
        res["silhouettes_numeric_only"] = {k: round(float(s), 3) for k, s in zip(ks, sil_num)}

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    axes[0].plot(range(1, len(cum) + 1), cum * 100, marker="o")
    axes[0].axhline(80, color="red", ls="--", lw=1)
    axes[0].set(title="PCA - cumulative variance explained", xlabel="components", ylabel="% variance")
    axes[1].plot(ks, inertia, marker="o", color="#d95f02")
    axes[1].set(title="Elbow method (KMeans inertia)", xlabel="k", ylabel="inertia")
    axes[2].plot(ks, sil, marker="o", color="#1b9e77", label="all features")
    if sil_num:
        axes[2].plot(ks, sil_num, marker="s", color="#7570b3", label="numeric only")
    if sil_num:
        axes[2].legend()
    axes[2].set(title="Silhouette score by k", xlabel="k", ylabel="silhouette")
    saver.save(fig, "clustering_readiness")

    p2 = PCA(n_components=2).fit_transform(sample)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(p2[:, 0], p2[:, 1], s=10, alpha=0.5, color="#3b7dd8")
    ax.set(title="Listeners projected on first 2 principal components", xlabel="PC1", ylabel="PC2")
    saver.save(fig, "pca_2d_projection")
    return res


def preview_clusters(df, scaled, num, k, out_dir, saver):
    """Quick KMeans check with the k the project asks for (default 3).
    This is a PREVIEW to confirm the structure - the official model is trained in train_model.py."""
    feats = [c for c in num if c in scaled.columns]
    Xs = scaled[feats]
    km = KMeans(n_clusters=k, n_init=25, random_state=42).fit(Xs)
    pca = PCA(n_components=2, random_state=42).fit(Xs)
    coords = pca.transform(Xs)
    load = pca.components_[0]
    if load.sum() < 0:               # orient PC1 so that "higher = more listening activity"
        coords[:, 0] *= -1
        load = -load
    pc1 = coords[:, 0]
    order = pd.Series(pc1).groupby(km.labels_).mean().sort_values().index.tolist()   # low -> high activity
    names = ["Casual Listener", "Music Explorer", "Heavy Listener"] if k == 3 else [f"Segment {i + 1}" for i in range(k)]
    remap = {old: names[new] for new, old in enumerate(order)}
    labels = pd.Series(km.labels_).map(remap)

    tbl = df[feats].groupby(labels.values).mean().loc[names].round(2)
    tbl.insert(0, "listeners", labels.value_counts().reindex(names).values)
    tbl.insert(1, "share_%", (tbl["listeners"] / len(df) * 100).round(1))
    out = df.copy()
    out["preview_segment"] = labels.values
    out.to_csv(os.path.join(out_dir, "cluster_preview_labels.csv"), index=False)

    colors = dict(zip(names, ["#4c9be8", "#f2a541", "#d64545"] if k == 3 else sns.color_palette("tab10", k).as_hex()))
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for n in names:
        m = labels.values == n
        ax.scatter(coords[m, 0], coords[m, 1], s=70, alpha=0.85, label=f"{n} ({m.sum()})", color=colors[n], edgecolor="white")
    ax.set(title=f"Preview: KMeans with k={k} on the PCA map", xlabel="PC1 (overall listening activity)", ylabel="PC2")
    ax.legend(loc="lower right")
    saver.save(fig, "preview_clusters_pca")

    fig, axes = plt.subplots(1, len(feats), figsize=(4 * len(feats), 4))
    axes = np.atleast_1d(axes)
    for ax, f in zip(axes, feats):
        vals = [tbl.loc[n, f] for n in names]
        bars = ax.bar(range(k), vals, color=[colors[n] for n in names])
        ax.set_xticks(range(k))
        ax.set_xticklabels([n.split()[0] for n in names], fontsize=9)
        ax.set_title(f.replace("_", " "), fontsize=10)
        ax.bar_label(bars, fmt="%.1f", fontsize=9)
    fig.suptitle("Preview: average value of each feature per cluster", y=1.03, fontsize=13)
    saver.save(fig, "preview_cluster_averages")

    return {"k": k, "table": tbl.reset_index().rename(columns={"index": "segment"}).to_dict(orient="records"),
            "pc1_variance": round(float(pca.explained_variance_ratio_[0]) * 100, 1),
            "pc1_loadings": {f: round(float(v), 3) for f, v in zip(feats, load)}}


# =============================================================== 5. findings
def write_findings(path, raw, df, rep, id_cols, num, cat, high_card, out_tbl, prep, prev, pre):
    c = rep["counts"]
    L = []
    sec = iter(range(1, 20))
    L += ["# Music Listener Segmentation - EDA Findings", "",
          "_Prepared by: Pavithra - Data Analysis & EDA Lead_", ""]

    L += ["## " + str(next(sec)) + ". Dataset overview",
          f"- Raw data: **{raw.shape[0]} rows x {raw.shape[1]} columns**; after cleaning: **{c['rows_after']} rows x {df.shape[1]} columns**",
          f"- ID column(s): {', '.join(id_cols) or 'none'}",
          f"- Numeric behaviour features ({len(num)}): {', '.join(num)}",
          f"- Categorical features ({len(cat)}): {', '.join(cat) or 'none'}"]
    if high_card:
        L.append(f"- High-cardinality text columns excluded from clustering: {', '.join(high_card)}")
    L.append("")

    L += ["## " + str(next(sec)) + ". Data quality issues found and fixed",
          f"- Duplicate rows removed: **{c['duplicate_rows_removed']}**; duplicate IDs removed: **{c['duplicate_ids_removed']}**",
          f"- Placeholder text (NA, null, '-', ...) turned into missing: **{c['placeholder_text_to_missing']}**",
          f"- Category spelling/case inconsistencies standardised: **{c['category_labels_standardised']}**"]
    if c["text_columns_converted_to_numeric"]:
        L.append(f"- Numbers stored as text converted: {', '.join(c['text_columns_converted_to_numeric'])}")
    if c["invalid_values_set_to_missing"]:
        L.append("- Invalid / impossible values (negative counts, rate outside 0-1, unrealistic age) replaced: "
                 + ", ".join(f"{k} ({v})" for k, v in c["invalid_values_set_to_missing"].items()))
    if c["missing_values_filled"]:
        L.append("- Missing values filled: " + ", ".join(f"{k} ({v})" for k, v in c["missing_values_filled"].items()))
    if c["columns_dropped_over_50pct_missing"]:
        L.append(f"- Columns dropped (>50% missing): {', '.join(c['columns_dropped_over_50pct_missing'])}")
    L.append("")

    if num:
        s = df[num].agg(["mean", "median", "std", "min", "max", "skew"]).T.round(2)
        s.index.name = "feature"
        L += ["## " + str(next(sec)) + ". Numeric feature summary", md_table(s), ""]
        skewed = [f"{k} ({v:.1f})" for k, v in df[num].skew().items() if abs(v) > 1]
        L.append("- Strongly skewed features (|skew| > 1): " + (", ".join(skewed) or "none"))
        heavy = out_tbl[out_tbl["outlier_%"] > 2]
        if len(heavy):
            L.append("- Features with >2% outliers (IQR rule): "
                     + ", ".join(f"{i} ({r['outlier_%']}%)" for i, r in heavy.iterrows()))
        L.append("")

    if cat:
        L.append("## " + str(next(sec)) + ". Categorical composition")
        for col in cat:
            vc = (df[col].value_counts(normalize=True) * 100).round(1)
            L.append(f"- **{col}**: " + ", ".join(f"{k} {v}%" for k, v in vc.head(5).items()))
        L.append("")

    if len(num) >= 2:
        corr = df[num].corr()
        pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
        top = pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(6)
        L += ["## " + str(next(sec)) + ". Strongest relationships between features"]
        for (a, b), r in top.items():
            tag = "positive" if r > 0 else "negative"
            L.append(f"- {a} <-> {b}: r = {r:.2f} ({tag})")
        red = [f"{a}/{b}" for (a, b), r in pairs.items() if abs(r) > 0.85]
        if red:
            L.append(f"- Highly correlated pairs (|r| > 0.85, possible redundancy for clustering): {', '.join(red)}")
        L.append("")

    if num and cat:
        L.append("## " + str(next(sec)) + ". Listening-behaviour patterns by group")
        z = (df[num] - df[num].mean()) / df[num].std().replace(0, 1)
        for col in cat:
            if not (2 <= df[col].nunique() <= 12):
                continue
            prof = z.groupby(df[col]).mean()
            spread = (prof.max() - prof.min()).sort_values(ascending=False).head(3)
            for feat in spread.index:
                hi, lo = prof[feat].idxmax(), prof[feat].idxmin()
                mean_hi = df.loc[df[col] == hi, feat].mean()
                mean_lo = df.loc[df[col] == lo, feat].mean()
                L.append(f"- By **{col}**: `{feat}` is highest for *{hi}* (avg {mean_hi:.2f}) and lowest for *{lo}* (avg {mean_lo:.2f})")
        L.append("")

    L += ["## " + str(next(sec)) + ". Clustering readiness",
          f"- Clustering-ready file: `music_listeners_clustering_ready.csv` ({len(prep['features'])} features)",
          f"- Outliers capped (IQR): {', '.join(prep['capped']) or 'none'}",
          f"- Log-transformed (skewed): {', '.join(prep['log_transformed']) or 'none'}",
          f"- One-hot encoded: {', '.join(prep['one_hot']) or 'none'}; numeric features standardised (mean 0, std 1)",
          f"- PCA: {prev['components_for_80pct']} components explain 80% of the variance; first 2 components explain {prev['pc1_pc2_variance']}%",
          f"- Silhouette (all features) is highest at **k = {prev['best_k_by_silhouette']}** (score {prev['best_silhouette']}); "
          f"scores by k: {prev['silhouettes']}"]
    if "best_k_numeric_only" in prev:
        L.append(f"- Silhouette (numeric features only) is highest at **k = {prev['best_k_numeric_only']}** "
                 f"(score {prev['best_silhouette_numeric_only']}); scores by k: {prev['silhouettes_numeric_only']}")
        if prev["best_k_numeric_only"] != prev["best_k_by_silhouette"]:
            L.append("- The two differ, so the one-hot categorical columns may be diluting the behaviour signal - "
                     "try clustering on numeric features only and use categories to describe the clusters afterwards.")
    L += ["- Suggested next step for the modelling teammate: try KMeans / Agglomerative with k between "
          f"{max(2, min(prev['best_k_by_silhouette'], prev.get('best_k_numeric_only', 99)) - 1)} and "
          f"{max(prev['best_k_by_silhouette'], prev.get('best_k_numeric_only', 0)) + 1} and compare with the elbow chart.", ""]

    L += ["## " + str(next(sec)) + ". Preview of the expected clusters (KMeans, k = %d)" % pre["k"],
          "_Quick check only - the official model is trained by the modelling teammate in train_model.py._", "",
          md_table(pd.DataFrame(pre["table"]), index=False), "",
          f"- PC1 alone explains **{pre['pc1_variance']}%** of the variation: the features move together along one 'listening activity' axis.", ""]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# =============================================================== main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="music_listeners.csv")
    ap.add_argument("--output", default="eda_output")
    ap.add_argument("--preview-k", type=int, default=3, help="clusters for the preview check")
    a = ap.parse_args()

    if not os.path.exists(a.input):
        raise SystemExit(f"'{a.input}' not found. Put the file in this folder or use --input <path>.\n"
                         "No file yet? Run 'python generate_sample_data.py' for practice data.")
    os.makedirs(a.output, exist_ok=True)
    saver = Saver(os.path.join(a.output, "charts"))

    print("1/5 Loading and inspecting ...")
    raw = load(a.input)
    with open(os.path.join(a.output, "data_overview.txt"), "w", encoding="utf-8") as f:
        f.write(inspect(raw))

    print("2/5 Cleaning and validating ...")
    df, rep = clean(raw)
    id_cols, num, cat, high_card = split_columns(df)
    out_tbl = outlier_table(df, num)
    df.to_csv(os.path.join(a.output, "music_listeners_cleaned.csv"), index=False)
    with open(os.path.join(a.output, "data_quality_report.txt"), "w", encoding="utf-8") as f:
        f.write("STEPS\n" + "\n".join(f"- {s}" for s in rep["steps"]) + "\n\nCOUNTS\n")
        f.write(json.dumps(rep["counts"], indent=2, default=str))
        f.write("\n\nOUTLIERS (IQR rule, kept in cleaned file, capped in clustering-ready file)\n")
        f.write(out_tbl.to_string())

    print("3/5 Creating EDA charts ...")
    eda_charts(raw, df, num, cat, saver)

    print("4/5 Preparing clustering-ready dataset ...")
    scaled, prep = prepare_for_clustering(df, id_cols, num, cat, a.output)
    prev = clustering_preview(scaled, saver, [c for c in scaled.columns if c in num])

    pre = preview_clusters(df, scaled, num, a.preview_k, a.output, saver)

    print("5/5 Writing findings.md ...")
    write_findings(os.path.join(a.output, "findings.md"), raw, df, rep, id_cols, num, cat, high_card, out_tbl, prep, prev, pre)
    results = {
        "rows": int(len(df)), "features": num,
        "describe": df[num].describe().T.round(3).reset_index().rename(columns={"index": "feature"}).to_dict(orient="records"),
        "skew": df[num].skew().round(3).to_dict(),
        "corr": df[num].corr().round(3).to_dict(),
        "outliers": out_tbl.reset_index().to_dict(orient="records"),
        "cleaning": rep["counts"], "prep": prep, "clustering_preview": prev, "cluster_preview": pre,
    }
    with open(os.path.join(a.output, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nDone. Everything is in '{a.output}/':")
    for name in sorted(os.listdir(a.output)):
        print("  -", name)


if __name__ == "__main__":
    main()
