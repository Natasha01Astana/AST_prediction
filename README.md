
# **AST Prediction** 📊🔬

**Predictive pipeline for serum AST (aspartate aminotransferase) levels using NHANES (1988–2018) clinical, biochemical, and lifestyle data.**  
_This project integrates data cleaning, feature engineering, ensemble modeling, interpretability (SHAP + robust correlations), clustering, calibration, classification, and mediation analysis._

---

## 🔗 **Data Access**

All raw and cleaned datasets used in this study are available in the shared Google Drive folder:  
**[NHANES Data for AST Prediction](https://drive.google.com/drive/folders/1cgyQXj3Kl7FdDoyPlmEkCKyDXNDIv4JB?usp=sharing)**

**Inside you will find:**
- **Demographics.csv** — _Age, sex, race, education, income, survey weights_
- **Response.csv** — _AST (LBXSASSI) and related laboratory measurements (ALT, ALP, LDH, Gamma-GT, etc.)_
- **Questionnaire.csv** — _Physical activity, alcohol consumption, smoking, medical history_
- **Chemical_Labs.csv** — _Ferritin, homocysteine, hs-CRP, cholesterol, glucose, uric acid, creatinine, hemoglobin_
- **Derived Indicators** — _Computed BMI, pseudotime clusters, binary AST_high labels_
- **Dictionaries & Metadata** — _Variable definitions, codebooks, cleaning logs_

> _Each module contains both “raw” and “cleaned” versions. Use the cleaned files for preprocessing._

---

## 🚀 **Project Overview**

### **1. Data Integration & Cleaning**
- Merge _demographics_, _response_, _questionnaire_, and _chemical lab_ modules on unique participant ID (`SEQN`).
- Filter AST values ≤ 200 U/L to exclude extreme outliers.
- Drop any row with missing values in selected features or target.

### **2. Feature Selection & Engineering**
- **Target variable:** `LBXSASSI` (_continuous AST_)
- **Predictors (X):**
  - **Demographics:**  
    - `RIDAGEYR` — *Age (years)*  
    - `RIAGENDR` — *Sex (1 = male, 2 = female)*
  - **Anthropometry:**  
    - `BMXWT` — *Weight (kg)*  
    - `BMXHT` — *Height (cm)*
  - **Biochemistry:**  
    - `LBXFER` — *Ferritin (μg/L)*  
    - `LBXHCY` — *Homocysteine (μmol/L)*  
    - `LBXTC` — *Total Cholesterol (mg/dL)*  
    - `LBDLDL` — *LDL Cholesterol (mg/dL)*  
    - `LBXGLU` — *Fasting Glucose (mg/dL)*  
    - `LBXHGB` — *Hemoglobin (g/dL)*  
    - `LBXSCR` — *Creatinine (mg/dL)*  
    - `LBXCRP` — *hs-CRP (mg/dL)*  
    - `LBXSAPSI` — *Alkaline Phosphatase (U/L)*  
    - `LBXSGTSI` — *Gamma-GT (U/L)*  
    - `LBXSLDSI` — *LDH (U/L)*  
    - `LBXSBU` — *BUN (mg/dL)*  
    - `LBXSUA` — *Uric Acid (mg/dL)*  
    - `LBXWBCSI` — *WBC (10³/µL)*
  - **Lifestyle & Behavior:**  
    - `ALQ130` — *Average daily alcohol intake (drinks/day)*  
    - `PAD615` — *Weekly minutes of moderate/vigorous activity*
  - **Derived Feature:**  
    - *BMI = BMXWT / (BMXHT ÷ 100)²*

### **3. Exploratory & Correlation Analysis**
- Compute _Pearson correlation_ and _Mutual Information_ for each predictor vs. AST.
- Calculate _robust associations_ (Spearman, Kendall, DistanceCorr, MICe) to capture nonlinear relationships.
- Visualize _feature importance_ & _correlation heatmap_.

### **4. Model Training & Evaluation**
#### **Regression Models:**
- Linear Regression  
- Random Forest (n_trees=300, max_depth=8)  
- XGBoost (300 trees, max_depth=8, lr=0.05)  
- CatBoost with Huber Loss (iterations=1200, depth=6, lr=0.03)  

#### **Stacked Ensembles:**
- **Stacking_v1:**  
  _Base learners: Linear, RF, XGB_  
  _Meta learner: Linear Regression_  
  _5‐fold CV, no passthrough_
- **Stacking_v2:**  
  _Base learners: CatBoost, LightGBM, ExtraTrees_  
  _Meta learner: Ridge_  
  _5‐fold CV, no passthrough_
- **Stacking_v3:**  
  _Two‐Stage High‐AST model (LGBM on AST > 50) + base learners: CatBoost (Huber), LGBM (Huber), XGB, RF, High‐AST submodel_  
  _Meta learner: LGBM (Huber) with passthrough features_  
  _5‐fold CV_

#### **Metrics (Train/Validation):**
- **R²** – Coefficient of determination  
- **RMSE** – Root mean squared error  
- **MAE** – Mean absolute error  
- **MAPE (%)** – Mean absolute percentage error  
- **Explained Variance**

*Visualize all metrics as grouped bar plots (Train vs Val).*

### **5. Interpretability & Feature Importance**
- **SHAP (XGBoost):**  
  _Beeswarm plot: distribution of SHAP values per feature_  
  _Mean |SHAP| ranking: global importance bar chart_
- **Permutation Importance (XGB)**
- **Robust Correlation Heatmap (Spearman, Kendall, DistanceCorr, MICe)**
- **Top‐10 SHAP feature interactions** (horizontal bar chart)

### **6. Hierarchical Clustering (Ward Linkage)**
- Compute _Pearson correlation matrix_ of (20 features + AST).
- Convert to pairwise distance: 1 – corr.
- Perform _Ward clustering_, draw dendrogram.
- **Clusters naturally separate:**
  - *Liver enzymes & cytolysis markers*
  - *Metabolic/inflammatory markers*
  - *Demographics & lifestyle*

### **7. Pseudotime Clustering (KMeans)**
- Normalize height (MinMax).
- Cluster (n_clusters=5) on (height_norm, AST).
- Order clusters by ascending mean(height_norm) → assign “pseudotime” index.
- Calculate average feature values per pseudotime cluster, compute deltas between consecutive steps.

### **8. Calibration Plot (LGBM)**
- Split test set into 10 prediction‐quantiles.
- Plot (mean_predicted vs mean_observed) for each bin + ideal diagonal.

### **9. Binary Classification: AST ≥ 40**
- Label `ast_high = 1` if AST ≥ 40, else 0.
- Train GradientBoostingClassifier on (X_train, ast_high).
- Evaluate: **ROC AUC** & **Average Precision (AP)**.
- Plot **ROC curve** and **Precision–Recall curve**.

### **10. Mediation Analysis (Pingouin)**
- Assess indirect vs direct effects of Ferritin (`LBXFER`) on AST via each mediator.
- **Mediation formula:**  
  - Path _a_: Ferritin → Mediator  
  - Path _b_: Mediator → AST (adjusting for Ferritin)  
  - Direct effect: c' path (Ferritin → AST adjusting for Mediator)  
  - Indirect effect: a × b  
  - Total effect: c = c' + (a × b)
- Repeat for all candidate mediators (_p = 21 → all features + BMI_).
- Output table with coef, SE, p‐value, 95% CI, significance for Direct/Indirect/Total.

---

## 🔧 **Installation & Quick Start**

```bash
# Clone repository
git clone https://github.com/<username>/AST_prediction.git
cd AST_prediction

# Install dependencies
pip install -r requirements.txt
````

* **Download data:**
  Place all CSV modules from the Google Drive link into `data/`.
  Ensure the following files exist:

  * `data/Demographics.csv`
  * `data/Response.csv`
  * `data/Questionnaire.csv`
  * `data/Chemical_Labs.csv`
  * `data/Dictionaries/`


* **Launch Jupyter Notebook:**
  Open and run sequentially the single notebook:
  `AST_Analysis_prediction.ipynb`


## 📈 **Key Findings**

* **Stacking\_v2** (*CatBoost + LightGBM + ExtraTrees → Ridge*) yields the highest validation **R²** (\~0.92) with minimal overfitting.
* **Top Predictors** (*by mean|SHAP| & robust correlations*):

  1. *Ferritin (LBXFER)*
  2. *Gamma‐GT (LBXSGTSI)*
  3. *LDH (LBXSLDSI)*
  4. *Fasting Glucose (LBXGLU)*
  5. *Hemoglobin (LBXHGB)*
* **Hierarchical Clustering** clearly groups:

  * *Liver/cytolysis enzymes* (Ferritin, Gamma‐GT, LDH, ALP)
  * *Metabolic & inflammatory markers* (Glucose, LDL, hs-CRP)
  * *Demographics & lifestyle* (Age, Sex, Alcohol, Physical Activity)
* **High‐AST Classification** (Gradient Boosting) achieves perfect separation on validation:

  * **ROC AUC = 1.000**
  * **AP (PR AUC) = 1.000**
* **Mediation Insights:**

  * *Ferritin’s effect on AST is mainly direct (p < 1e-08).*
  * *Indirect paths through Gamma-GT, LDH, Glucose, BUN, and Uric Acid are statistically significant, highlighting multi‐pathway regulation.*

> *These results support a multifactorial framework: iron metabolism, hepatocellular injury, and cardiometabolic risk all interplay to shape AST variability in the general population.*


