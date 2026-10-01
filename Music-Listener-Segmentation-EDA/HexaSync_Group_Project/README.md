# Team Roles & Responsibilities

This project consists of two Machine Learning projects:

1. **Advanced Loan Approval Predictor** — Supervised Machine Learning
2. **Music Listener Segmentation** — Unsupervised Machine Learning



---

## 🏦 Project 1 — Advanced Loan Approval Predictor

**Machine Learning Type:** Supervised Learning

### 1. Mugilan — ML & Model Development Lead

**Responsibilities:**

* Develop and maintain the main ML training pipeline.
* Perform data preprocessing and feature engineering.
* Train suitable classification models.
* Compare model performance.
* Evaluate models using appropriate evaluation metrics.
* Select the final machine learning model.
* Save and load the trained model and scaler using Joblib.
* Support final model integration.
* Explain the machine learning implementation during the project demonstration.

**Primary Area:**
`train_model.py`

---

### 2. Ashaaz Ahmed Khan — Dataset & Testing Lead

**Responsibilities:**

* Work with `loans.csv`.
* Develop and maintain `dataset.py` as required.
* Handle dataset loading and validation.
* Understand and document dataset features and target variables.
* Check data quality and preprocessing requirements.
* Prepare test cases for different applicant inputs.
* Perform final testing of the complete application.
* Record testing results and identify issues.
* Assist with final project validation.

**Primary Areas:**
`dataset.py`
`loans.csv`
Testing & Validation

---

### 3. Anto Roshan — Backend & Frontend Developer

**Responsibilities:**

* Develop the backend API using FastAPI.
* Implement `main.py`.
* Create the prediction API endpoint.
* Load the trained model and scaler.
* Connect the API with the machine learning model.
* Develop the Streamlit frontend using `app.py`.
* Connect the Streamlit frontend with the FastAPI backend.
* Create the applicant input form.
* Display prediction results clearly.
* Test frontend and backend communication.

**Primary Areas:**
`main.py`
`app.py`

---

# 🎵 Project 2 — Music Listener Segmentation

**Machine Learning Type:** Unsupervised Learning

### 4. Pavithra — Data Analysis & EDA Lead

**Responsibilities:**

* Work with `music_listeners.csv`.
* Understand the listener behaviour features.
* Perform data cleaning and validation.
* Conduct Exploratory Data Analysis (EDA).
* Analyze listening behaviour patterns.
* Create relevant charts and visualizations.
* Prepare the dataset for clustering.
* Document important findings from the analysis.

**Primary Area:**
`music_listeners.csv`
Data Analysis & EDA

---

### 5. Srimathi — Clustering & ML Lead

**Responsibilities:**

* Develop and maintain `train_model.py`.
* Select the required clustering features.
* Apply feature scaling using `StandardScaler`.
* Implement K-Means clustering.
* Analyze the resulting cluster centers.
* Interpret the three listener groups.
* Assign meaningful listener segment names.
* Save the trained K-Means model as `model.pkl`.
* Save the scaler as `scaler.pkl`.
* Add Elbow Method and/or Silhouette analysis where appropriate.

**Primary Area:**
`train_model.py`

---

### 6. Nithish — Application & UI Lead

**Responsibilities:**

* Develop the application interface.
* Implement `app.py`.
* Load the trained `model.pkl` and `scaler.pkl`.
* Create the listener input interface.
* Apply the saved scaler using `transform()`.
* Predict the listener's cluster.
* Display the corresponding listener segment.
* Improve the application's UI and visual output.
* Test the application with different listener profiles.

**Primary Area:**
`app.py`


