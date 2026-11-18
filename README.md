# 🚀 IMDB Movie Rating Prediction & Analysis

This repository contains our machine learning project analyzing and predicting film ratings using the **IMDB Top 1000** dataset.

---

## 👥 Group Members & Assigned Roles
* **Role 1: Data Manager (Hammad)** — Data acquisition, repository setup, EDA, and basic cleaning
* **Role 2: Preprocessing Specialist (Sabrina)** — Feature engineering, categorical encoding, feature scaling, and data leakage prevention
* **Role 3: Model Architect (Novie)** — Model selection, hyperparameter tuning setup, and GridSearchCV/RandomizedSearchCV pipelines
* **Role 4: Evaluation Analyst & Presenter (Jorge)** — Performance metrics, comparative plots, error analysis, and presentation coordination

---

## 🎯 Problem Statement
Streaming platforms and film analysts want to know **which measurable traits of a film are associated with higher audience ratings**. Acting as data science consultants, we predict a film's IMDB rating from its critical reception, audience reach, box office, runtime, era, age certificate, genres, and top creators/cast.

* **Task Type:** Regression (predicting continuous ratings)
* **Target Variable:** `IMDB_Rating` (ranging from 7.6 to 9.3)
* **Input Features:** 57 engineered features (continuous, scaled variables, one-hot encoded genres, and top-performing creators)

---

## 📊 Preprocessing & Feature Engineering Pipeline
1. **Multi-Genre Handling:** One-hot encoded 21 distinct genres and calculated the number of genres per film.
2. **High-Cardinality Categoricals:** Extracted and engineered binary features for top-15 directors and top-15 lead actors.
3. **Data Leakage Mitigation:** Splitting data into an 80/20 train/test split before fitting any transformations.
4. **Imputation:** Handling missing values in continuous variables utilizing training-set medians.
5. **Standardization:** Robust feature scaling via `StandardScaler` applied strictly to continuous variables.

---

## 🏗️ Models to Compare
We set up a progressive complexity analysis using the following models tuned via `GridSearchCV` with **10-fold cross-validation**:
* **Ridge Regression** (Linear regularization)
* **Decision Tree** (Instability and flowchart analysis)
* **Random Forest** (Ensemble averaging over multiple trees)
