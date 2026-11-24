# Smart Crop & Agriculture Analyzer 🌾

An end-to-end Machine Learning project for precision agriculture and decision support, structured **exclusively as Jupyter Notebooks (`.ipynb`)**.

This is a college-level AI/ML academic project demonstrating core traditional Machine Learning concepts across:
1. **Multiclass Classification** (Crop Recommendation)
2. **Regression** (Crop Yield Prediction)
3. **Unsupervised Clustering** (Soil Pattern Analysis)

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Project Objectives](#3-project-objectives)
4. [Technologies Used](#4-technologies-used)
5. [Datasets & Sources](#5-datasets--sources)
6. [Machine Learning Fundamentals: Supervised vs. Unsupervised](#6-machine-learning-fundamentals-supervised-vs-unsupervised)
7. [ML Modules & Algorithmic Breakdown](#7-ml-modules--algorithmic-breakdown)
   - [Module A: Crop Recommendation (Classification)](#module-a-crop-recommendation-classification)
   - [Module B: Crop Yield Prediction (Regression)](#module-b-crop-yield-prediction-regression)
   - [Module C: Soil Pattern Analysis (Clustering)](#module-c-soil-pattern-analysis-clustering)
8. [Data Preprocessing & Hygiene](#8-data-preprocessing--hygiene)
9. [Train-Test Splitting & Preventing Data Leakage](#9-train-test-splitting--preventing-data-leakage)
10. [Model Evaluation Metrics](#10-model-evaluation-metrics)
11. [Jupyter Notebooks Structure](#11-jupyter-notebooks-structure)
12. [Installation & Setup](#12-installation--setup)
13. [How to Run the Jupyter Notebooks](#13-how-to-run-the-jupyter-notebooks)
14. [Sample Inputs and Outputs](#14-sample-inputs-and-outputs)
15. [Limitations and Future Scope](#15-limitations-and-future-scope)
16. [College Viva Voce Q&A Cheat Sheet](#16-college-viva-voce-qa-cheat-sheet)

---

## 1. Project Overview

Agriculture is deeply dependent on soil chemistry, weather conditions, and seasonal timing. Traditional farming often relies on guesswork or generalized rules of thumb, leading to suboptimal crop selection, fertilizer misuse, and poor harvest yields.

**Smart Crop & Agriculture Analyzer** applies established Machine Learning algorithms to historical agricultural records to:
- Recommend the best crop for a given plot of land based on soil NPK, pH, and local weather.
- Predict expected harvest productivity in tonnes per hectare.
- Cluster soil samples into distinct agronomic profiles using unsupervised learning.

---

## 2. Problem Statement

Farmers face three critical questions every agricultural cycle:
1. *Which crop has the highest biological likelihood of thriving under my soil and climate conditions?*
2. *What harvest yield can I expect per hectare for my selected crop, season, and state?*
3. *What natural nutrient group does my soil fall into, and what does that imply for fertilizer management?*

This project solves all three problems through a unified, interactive Jupyter Notebook workflow without relying on black-box neural networks or paid APIs.

---

## 3. Project Objectives

- Implement **Multiclass Classification** comparing Random Forest, Decision Tree, and K-Nearest Neighbors (KNN).
- Implement **Regression** comparing Linear Regression and Random Forest Regressor on district-level Indian agricultural data.
- Implement **Unsupervised Clustering** using K-Means (k=4) with 2D Principal Component Analysis (PCA) for visualization.
- Ensure 100% reproducible test-set metrics without hardcoded numbers.
- Provide clean, cell-by-cell executable Jupyter Notebooks suitable for college viva demonstrations.

---

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| **Python 3.10+** | Core programming language |
| **Jupyter Notebook (`.ipynb`)** | Interactive notebook development & execution format |
| **Pandas** | Tabular data manipulation and CSV handling |
| **NumPy** | Array transformations and mathematical calculations |
| **Scikit-Learn** | Machine Learning models, pipelines, preprocessing, and metrics |
| **Matplotlib & Seaborn** | Evaluation charts, confusion matrices, and PCA scatter plots |
| **Joblib** | Model serialization and deserialization (`.pkl`) |

*Note: In accordance with project requirements, no deep learning frameworks (TensorFlow, PyTorch) or external LLM APIs are used.*

---

## 5. Datasets & Sources

This project uses authentic public datasets focusing on Indian agriculture:

### Dataset 1: Crop Recommendation Dataset
- **File:** `data/crop_recommendation.csv`
- **Size:** 2,200 records, 8 columns
- **Features:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, Soil pH, Rainfall
- **Target:** 22 Crop Classes (`rice`, `maize`, `chickpea`, `kidneybeans`, `pigeonpeas`, `mothbeans`, `mungbean`, `blackgram`, `lentil`, `pomegranate`, `banana`, `mango`, `grapes`, `watermelon`, `muskmelon`, `apple`, `orange`, `papaya`, `coconut`, `cotton`, `jute`, `coffee`)
- **Distribution:** Balanced (100 records per crop class)
- **Source Link:** [Hugging Face / GitHub Precision Agriculture Dataset](https://github.com/PAIshanMadusha/crop-recommendation-model/blob/main/dataset/Crop_recommendation.csv)

### Dataset 2: Indian District-Level Crop Production Dataset
- **File:** `data/crop_yield.csv`
- **Size:** 246,091 raw records (cleaned to 227,008 records)
- **Features:** `Crop`, `Season`, `State_Name`, `Area` (hectares)
- **Target:** `Yield` = `Production (tonnes)` / `Area (hectares)`
- **Coverage:** 33 Indian States/UTs, 6 Seasons, 46 Major Crops
- **Source Link:** Ministry of Agriculture & Farmers Welfare, Government of India (Public domain via [Kaggle](https://www.kaggle.com/datasets/abhinand05/crop-production-in-india) and [data.gov.in](https://data.gov.in/))

---

## 6. Machine Learning Fundamentals: Supervised vs. Unsupervised

| Property | Supervised Learning | Unsupervised Learning |
|---|---|---|
| **Data Type** | Labeled data $(X, y)$ with known targets | Unlabeled data $(X)$ with no ground-truth labels |
| **Objective** | Learn a mapping function $f(X) \approx y$ | Discover underlying clusters, patterns, or density |
| **Modules Used** | Module A (Classification), Module B (Regression) | Module C (K-Means Soil Clustering) |
| **Feedback** | Direct error evaluation against true test labels | Intrinsic metrics like Silhouette Score |

---

## 7. ML Modules & Algorithmic Breakdown

### Module A: Crop Recommendation (Classification)
- **Task:** Multiclass Supervised Classification
- **Algorithms Compared:**
  1. **Random Forest Classifier:** An ensemble of 100 decision trees trained using bootstrap aggregation (bagging) and feature sub-sampling. Achieved **99.55% test accuracy**, virtually eliminating variance and overfitting.
  2. **Decision Tree Classifier:** Builds a hierarchical flowchart of binary decisions based on Gini Impurity. Achieved **97.95% accuracy**.
  3. **K-Nearest Neighbors (KNN):** Distance-based instance learner with $k=5$ and Euclidean distance. Features are scaled via `StandardScaler` to prevent high-magnitude features (e.g., rainfall) from dominating. Achieved **98.18% accuracy**.

**Selected Model:** `Random Forest Classifier` (Saved as `models/crop_classifier.pkl`).

---

### Module B: Crop Yield Prediction (Regression)
- **Task:** Continuous Value Prediction
- **Target:** Yield in Tonnes per Hectare ($\text{t/ha}$)
- **Algorithms Compared:**
  1. **Linear Regression:** Fits an optimal hyperplane minimizing Ordinary Least Squares (OLS) residual sum of squares:
     $$\min_{\beta} \|y - X\beta\|_2^2$$
     Achieved $R^2 = 0.7687$, $\text{MAE} = 2.076\text{ t/ha}$.
  2. **Random Forest Regressor:** Non-linear ensemble of 50 decision trees capturing multi-way interactions between crop types, seasonal rainfall patterns, regional soil climates, and farm acreage.
     Achieved **$R^2 = 0.9083$**, **$\text{MAE} = 1.1273\text{ t/ha}$**.

**Selected Model:** `Random Forest Regressor Pipeline` (Saved as `models/yield_regressor.pkl`).

---

### Module C: Soil Pattern Analysis (Clustering)
- **Task:** Unsupervised Clustering
- **Features Used:** Numerical soil chemistry only: `['N', 'P', 'K', 'ph']`.
- **Algorithm:** K-Means with $k=4$ and `k-means++` initialization.
- **Silhouette Score:** `0.3826` (well-separated clusters).
- **Visualization:** 2D Principal Component Analysis (PCA) projection.
- **Cluster Profiles:**
  - **Cluster 0 (Legume & Neutral Soil):** Moderate N (~32), moderate P (~53), K (~33), neutral-alkaline pH (~7.2). Optimal for nitrogen-fixing pulses (chickpea, lentil, blackgram).
  - **Cluster 1 (High Nitrogen Soil):** High N (~96), moderate P (~42), K (~38), slightly acidic pH (~6.5). Supports heavy vegetative feeders (rice, maize, cotton, banana).
  - **Cluster 2 (Acidic Plantation Soil):** Low N (~22), low K (~26), acidic pH (~5.8). Well-suited for acid-tolerant plantation trees (coconut, mango).
  - **Cluster 3 (High P & K Orchard Soil):** Extremely high P (~133) and K (~200). Typical for fruit crops (apples, grapes) demanding heavy potassium for sugar synthesis.

---

## 8. Data Preprocessing & Hygiene

1. **Handling Missing Values:** Null values in target production were dropped (no synthetic guessing).
2. **Unit Consistency:** Raw agricultural statistics measure Coconut in count of nuts rather than metric tonnes. Mixing nuts with tonnes distorts crop yield; non-tonne crops were cleanly filtered.
3. **Outlier Filtering:** Government records occasionally contain scale typos (e.g., recording tonnes as kilograms, resulting in impossible yields of 80,000 t/ha). Realistic agronomic bounds ($0 < \text{Yield} \le 100\text{ t/ha}$) were applied.
4. **Encoding Categoricals:** One-Hot Encoding (`OneHotEncoder(handle_unknown='ignore')`) converted `Crop`, `Season`, and `State_Name` into numerical binary vectors.
5. **Feature Scaling:** `StandardScaler` was applied within pipelines for distance-sensitive models (KNN and K-Means).

---

## 9. Train-Test Splitting & Preventing Data Leakage

- **Split Ratio:** 80% Training, 20% Testing (`random_state=42` for strict reproducibility).
- **Stratified Split:** Applied to the classification dataset so every crop category has an identical 20% test representation (20 test samples per crop).
- **Leakage Prevention:** Transformers (scalers, encoders) were fitted **only** on the training set (`fit_transform`) and subsequently applied to test data (`transform`). Preprocessing objects are serialized inside Scikit-Learn `Pipeline` objects.

---

## 10. Model Evaluation Metrics

### Classification Metrics (Module A)
- **Accuracy:** Proportion of correct crop predictions over all test samples ($99.55\%$).
- **Precision:** $\frac{TP}{TP + FP}$ — proportion of predicted crops that were truly that crop.
- **Recall:** $\frac{TP}{TP + FN}$ — proportion of actual crops correctly detected.
- **F1-Score:** Harmonic mean of precision and recall.
- **Confusion Matrix:** $22 \times 22$ matrix showing exact ground-truth vs. predicted categories.

### Regression Metrics (Module B)
- **MAE (Mean Absolute Error):** Average magnitude of errors ($1.1273\text{ t/ha}$).
- **MSE (Mean Squared Error):** Penalizes larger prediction errors ($10.69$).
- **RMSE (Root Mean Squared Error):** $\sqrt{\text{MSE}} = 3.2699\text{ t/ha}$.
- **$R^2$ Score (Coefficient of Determination):** Proportion of variance in crop yield explained by the model ($90.83\%$).

### Clustering Metrics (Module C)
- **Silhouette Score:** Evaluates how similar an object is to its own cluster compared to other clusters ($s \in [-1, +1]$; achieved $+0.3826$).

---

## 11. Jupyter Notebooks Structure

All project files are formatted as Jupyter Notebooks (`.ipynb`) without any `.py` script files:

```
d:\ML_Projects\Crop_Analyzer\
│
├── Smart_Crop_and_Agriculture_Analyzer.ipynb  # MASTER NOTEBOOK: Complete end-to-end ML project
├── 01_Crop_Recommendation_Classification.ipynb # Module A: Crop Classification notebook
├── 02_Crop_Yield_Prediction_Regression.ipynb    # Module B: Crop Yield Regression notebook
├── 03_Soil_Pattern_Clustering.ipynb            # Module C: Unsupervised Soil Clustering notebook
│
├── data/
│   ├── crop_recommendation.csv                  # Crop recommendation dataset (2,200 records)
│   └── crop_yield.csv                           # Indian crop yield dataset (246k records)
│
├── models/
│   ├── crop_classifier.pkl                      # Best trained Random Forest classifier
│   ├── yield_regressor.pkl                      # Best trained Random Forest regressor pipeline
│   └── soil_cluster_model.pkl                   # K-Means model, StandardScaler, PCA & profiles
│
├── requirements.txt                            # Python dependencies
└── README.md                                   # Comprehensive documentation & viva guide
```

---

## 12. Installation & Setup

1. **Navigate to the Project Directory:**
   ```bash
   cd d:\ML_Projects\Crop_Analyzer
   ```

2. **Install Required Libraries:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 13. How to Run the Jupyter Notebooks

### Option A: In VS Code (Recommended)
1. Open the project folder `d:\ML_Projects\Crop_Analyzer` in VS Code.
2. Click on [Smart_Crop_and_Agriculture_Analyzer.ipynb](file:///d:/ML_Projects/Crop_Analyzer/Smart_Crop_and_Agriculture_Analyzer.ipynb).
3. Select your Python 3 kernel in the top-right corner.
4. Click **Run All** or execute cells individually with `Shift + Enter`.

### Option B: In Classic Jupyter Notebook
1. In your terminal, run:
   ```bash
   jupyter notebook
   ```
2. Your browser will open the directory view. Click on `Smart_Crop_and_Agriculture_Analyzer.ipynb`.
3. In the menu, click **Kernel** → **Restart & Run All**.

### Option C: In JupyterLab
```bash
jupyter lab
```

---

## 14. Sample Inputs and Outputs

### Test Sample 1: Crop Recommendation
- **Inputs:** $N=90$, $P=42$, $K=43$, $\text{Temp}=21.0\ ^\circ\text{C}$, $\text{Humidity}=82.0\%$, $\text{pH}=6.5$, $\text{Rainfall}=203.0\text{ mm}$
- **Output:** `RICE` (Confidence: 95.7%)
- **Agronomic Explanation:** High humidity, high rainfall (>200 mm), and balanced soil pH are optimal conditions for wetland rice cultivation.

### Test Sample 2: Crop Yield Prediction
- **Inputs:** Crop: `Wheat`, Season: `Rabi`, State: `Punjab`, Area: `10.0 Hectares`
- **Output:** `4.52 Tonnes / Hectare`
- **Total Estimated Harvest:** `45.20 Tonnes` across 10 hectares.

### Test Sample 3: Soil Pattern Clustering
- **Inputs:** $N=25$, $P=135$, $K=200$, $\text{pH}=6.0$
- **Assigned Cluster:** `Cluster 3: High Phosphorus & Potassium Orchard Soil Pattern`
- **Explanation:** Characterized by heavy potassium and phosphorus levels required for fruiting crops like apples and grapes.

---

## 15. Limitations and Future Scope

### Current Limitations:
1. Soil datasets do not account for micronutrients (Zinc, Boron, Iron, Manganese).
2. Weather inputs represent average seasonal values rather than real-time daily forecasts.
3. Market pricing, minimum support prices (MSP), and profit optimization are outside the current ML scope.

### Future Scope:
1. Integration with real-time Indian Meteorological Department (IMD) weather APIs.
2. Addition of crop disease recognition using computer vision.
3. Fertilizer dosage recommendation based on targeted yield equations.

---

## 16. College Viva Voce Q&A Cheat Sheet

### Q1: Why did you choose Random Forest over single Decision Trees?
> **Answer:** A single Decision Tree has high variance and easily overfits the training set by creating overly complex boundaries. Random Forest uses *bagging* (Bootstrap Aggregation) by training multiple independent trees on random subsets of data and features. Averaging their predictions significantly reduces variance and yields superior generalization (99.55% accuracy on our test set).

### Q2: Why is feature scaling essential for KNN and K-Means, but not for Decision Trees?
> **Answer:** KNN and K-Means rely on Euclidean distances:
> $$d(p, q) = \sqrt{\sum_{i=1}^n (p_i - q_i)^2}$$
> If one feature (e.g. Rainfall = 250 mm) has a much larger numerical range than another (e.g. pH = 6.5), the distance is dominated solely by rainfall. `StandardScaler` centers features to $\mu=0, \sigma=1$. Decision Trees, by contrast, evaluate split conditions feature-by-feature independently and are completely invariant to monotonic feature scaling.

### Q3: What is the difference between $R^2$ Score and RMSE in Regression?
> **Answer:** RMSE (Root Mean Squared Error) measures the absolute error magnitude in the original target units ($\text{tonnes/ha}$). $R^2$ (Coefficient of Determination) is a scale-independent metric indicating the proportion of variance in the target variable explained by the features ($0.0 \le R^2 \le 1.0$). Our Random Forest Regressor achieves $R^2 = 0.9083$, meaning it explains ~91% of yield variance.

### Q4: How did you calculate yield from the Indian agricultural dataset?
> **Answer:** The raw government dataset provides total `Production` (tonnes) and cultivated `Area` (hectares). Yield is mathematically derived as:
> $$\text{Yield} = \frac{\text{Production}}{\text{Area}} \quad (\text{tonnes per hectare})$$
> Records with zero area, null production, and non-tonne crops (such as Coconut, which is recorded in count of nuts) were removed to ensure valid unit consistency.

### Q5: How does K-Means determine the cluster for a new user sample?
> **Answer:** K-Means standardizes the user's input using the pre-fitted scaler, calculates the Euclidean distance between the scaled sample and the $k$ cluster centroids, and assigns the sample to the cluster whose centroid is geometrically closest.
