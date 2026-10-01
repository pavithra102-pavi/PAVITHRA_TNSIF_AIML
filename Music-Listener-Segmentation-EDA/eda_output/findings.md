# Music Listener Segmentation - EDA Findings

_Prepared by: Pavithra - Data Analysis & EDA Lead_

## 1. Dataset overview
- Raw data: **30 rows x 4 columns**; after cleaning: **30 rows x 4 columns**
- ID column(s): none
- Numeric behaviour features (4): listening_hours_per_week, songs_per_day, skip_rate, playlist_count
- Categorical features (0): none

## 2. Data quality issues found and fixed
- Duplicate rows removed: **0**; duplicate IDs removed: **0**
- Placeholder text (NA, null, '-', ...) turned into missing: **0**
- Category spelling/case inconsistencies standardised: **0**

## 3. Numeric feature summary
| feature | mean | median | std | min | max | skew |
|---|---|---|---|---|---|---|
| listening_hours_per_week | 15.17 | 12.5 | 10.8 | 2.0 | 38.0 | 0.64 |
| songs_per_day | 60.43 | 49.0 | 42.91 | 10.0 | 145.0 | 0.62 |
| skip_rate | 29.47 | 29.0 | 18.5 | 3.0 | 65.0 | 0.24 |
| playlist_count | 14.97 | 14.5 | 10.77 | 2.0 | 40.0 | 0.62 |

- Strongly skewed features (|skew| > 1): none

## 4. Strongest relationships between features
- listening_hours_per_week <-> songs_per_day: r = 1.00 (positive)
- listening_hours_per_week <-> playlist_count: r = 0.97 (positive)
- songs_per_day <-> playlist_count: r = 0.97 (positive)
- listening_hours_per_week <-> skip_rate: r = -0.93 (negative)
- songs_per_day <-> skip_rate: r = -0.93 (negative)
- skip_rate <-> playlist_count: r = -0.92 (negative)
- Highly correlated pairs (|r| > 0.85, possible redundancy for clustering): listening_hours_per_week/songs_per_day, listening_hours_per_week/skip_rate, listening_hours_per_week/playlist_count, songs_per_day/skip_rate, songs_per_day/playlist_count, skip_rate/playlist_count

## 5. Clustering readiness
- Clustering-ready file: `music_listeners_clustering_ready.csv` (4 features)
- Outliers capped (IQR): none
- Log-transformed (skewed): none
- One-hot encoded: none; numeric features standardised (mean 0, std 1)
- PCA: 1 components explain 80% of the variance; first 2 components explain 99.0%
- Silhouette (all features) is highest at **k = 2** (score 0.578); scores by k: {2: 0.578, 3: 0.553, 4: 0.529, 5: 0.481, 6: 0.477, 7: 0.442, 8: 0.434, 9: 0.429, 10: 0.412}
- Suggested next step for the modelling teammate: try KMeans / Agglomerative with k between 2 and 3 and compare with the elbow chart.

## 6. Preview of the expected clusters (KMeans, k = 3)
_Quick check only - the official model is trained by the modelling teammate in train_model.py._

| segment | listeners | share_% | listening_hours_per_week | songs_per_day | skip_rate | playlist_count |
|---|---|---|---|---|---|---|
| Casual Listener | 11 | 36.7 | 4.82 | 19.09 | 49.82 | 4.09 |
| Music Explorer | 11 | 36.7 | 14.45 | 58.0 | 24.91 | 15.64 |
| Heavy Listener | 8 | 26.7 | 30.38 | 120.62 | 7.75 | 29.0 |

- PC1 alone explains **96.7%** of the variation: the features move together along one 'listening activity' axis.
