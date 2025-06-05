AST Prediction 📊🔬
Predictive pipeline for serum AST (aspartate aminotransferase) levels using NHANES (1988–2018) clinical, biochemical, and lifestyle data. This project integrates data cleaning, feature engineering, ensemble modeling, interpretability (SHAP + robust correlations), clustering, calibration, classification, and mediation analysis.

🔗 Data Access
All raw and cleaned datasets used in this study are available in the shared Google Drive folder:
NHANES Data for AST Prediction

Inside you will find:

Demographics.csv – Age, sex, race, education, income, survey weights.

Response.csv – AST (LBXSASSI) and related laboratory measurements (ALT, ALP, LDH, Gamma-GT, etc.).

Questionnaire.csv – Physical activity, alcohol consumption, smoking, medical history.

Chemical_Labs.csv – Ferritin, homocysteine, hs-CRP, cholesterol, glucose, uric acid, creatinine, hemoglobin.

Derived Indicators – Computed BMI, pseudotime clusters, binary AST_high labels.

Dictionaries & Metadata – Variable definitions, codebooks, cleaning logs.

Each module contains both “raw” and “cleaned” versions. Use the cleaned files for preprocessing.

🚀 Project Overview
Data Integration & Cleaning
• Merge demographics, response, questionnaire, and chemical lab modules on unique participant ID (SEQN).
• Filter AST values ≤ 200 U/L to exclude extreme outliers.
• Drop any row with missing values in selected features or target.

Feature Selection & Engineering
• Target variable (y): LBXSASSI (continuous AST).
• Predictors (X):

Demographics:
• RIDAGEYR – Age (years)
• RIAGENDR – Sex (1 = male, 2 = female)

Anthropometry:
• BMXWT – Weight (kg)
• BMXHT – Height (cm)

Biochemistry:
• LBXFER – Ferritin (μg/L)
• LBXHCY – Homocysteine (μmol/L)
• LBXTC – Total Cholesterol (mg/dL)
• LBDLDL – LDL Cholesterol (mg/dL)
• LBXGLU – Fasting Glucose (mg/dL)
• LBXHGB – Hemoglobin (g/dL)
• LBXSCR – Creatinine (mg/dL)
• LBXCRP – hs-CRP (mg/dL)
• LBXSAPSI – Alkaline Phosphatase (U/L)
• LBXSGTSI – Gamma-GT (U/L)
• LBXSLDSI – LDH (U/L)
• LBXSBU – BUN (mg/dL)
• LBXSUA – Uric Acid (mg/dL)
• LBXWBCSI – WBC (10³/µL)

Lifestyle & Behavior:
• ALQ130 – Average daily alcohol intake (drinks/day)
• PAD615 – Weekly minutes of moderate/vigorous activity

• Derived Feature:
• BMI = BMXWT / (BMXHT ÷ 100)²

Exploratory & Correlation Analysis
• Compute Pearson correlation and Mutual Information for each predictor vs. AST.
• Calculate robust associations (Spearman, Kendall, DistanceCorr, MICe) to capture nonlinear relationships.
• Visualize feature importance & correlation heatmap.

Model Training & Evaluation
Regression Models:
• Linear Regression
• Random Forest (n_trees=300, max_depth=8)
• XGBoost (300 trees, max_depth=8, lr=0.05)
• CatBoost with Huber Loss (iterations=1200, depth=6, lr=0.03)
Stacked Ensembles:
• Stacking_v1
– Base learners: Linear, RF, XGB
– Meta learner: Linear Regression
– 5‐fold CV, no passthrough
• Stacking_v2
– Base learners: CatBoost, LightGBM, ExtraTrees
– Meta learner: Ridge
– 5‐fold CV, no passthrough
• Stacking_v3
– Two‐Stage High‐AST model (LGBM on AST > 50) + base learners: CatBoost (Huber), LGBM (Huber), XGB, RF, High‐AST submodel
– Meta learner: LGBM (Huber) with passthrough features
– 5‐fold CV

Metrics (Train/Validation):
• R² – Coefficient of determination
• RMSE – Root mean squared error
• MAE – Mean absolute error
• MAPE (%) – Mean absolute percentage error
• Explained Variance

Visualize all metrics as grouped bar plots (Train vs Val).

Interpretability & Feature Importance
• SHAP (XGBoost)
– Beeswarm plot: distribution of SHAP values per feature
– Mean |SHAP| ranking: global importance bar chart
• Permutation Importance (XGB)
• Robust Correlation Heatmap (Spearman, Kendall, DistanceCorr, MICe)
• Top‐10 SHAP feature interactions (horizontal bar chart)

Hierarchical Clustering (Ward Linkage)
• Compute Pearson correlation matrix of (20 features + AST).
• Convert to pairwise distance: 1 – corr.
• Perform Ward clustering, draw dendrogram.
• Clusters naturally separate:

Liver enzymes & cytolysis markers

Metabolic/inflammatory markers

Demographics & lifestyle

Pseudotime Clustering (KMeans)
• Normalize height (MinMax).
• Cluster (n_clusters=5) on (height_norm, AST).
• Order clusters by ascending mean(height_norm) → assign “pseudotime” index.
• Calculate average feature values per pseudotime cluster, compute deltas between consecutive steps.

Calibration Plot (LGBM)
• Split test set into 10 prediction‐quantiles.
• Plot (mean_predicted vs mean_observed) for each bin + ideal diagonal.

Binary Classification: AST ≥ 40
• Label ast_high = 1 if AST ≥ 40, else 0.
• Train GradientBoostingClassifier on (X_train, ast_high).
• Evaluate: ROC AUC & Average Precision (AP).
• Plot ROC curve and Precision–Recall curve.

Mediation Analysis (Pingouin)
• Assess indirect vs direct effects of Ferritin (LBXFER) on AST via each mediator.
• Mediation formula:
– Path a: Ferritin → Mediator
– Path b: Mediator → AST (adjusting for Ferritin)
– Direct effect: c' path (Ferritin → AST adjusting for Mediator)
– Indirect effect: a × b
– Total effect: c = c' + (a × b)
• Repeat for all candidate mediators (p = 21 → all features + BMI)
• Output table with coef, SE, p‐value, 95% CI, significance for Direct/Indirect/Total.


🔧 Installation & Quick Start
Clone repository

bash
Копировать
Редактировать
git clone https://github.com/<username>/AST_prediction.git
cd AST_prediction
Install dependencies

nginx
Копировать
Редактировать
pip install -r requirements.txt
Download data

Place all CSV modules from the Google Drive link into data/.

Ensure the following files exist:
• data/Demographics.csv
• data/Response.csv
• data/Questionnaire.csv
• data/Chemical_Labs.csv
• data/Dictionaries/

Run Preprocessing

css
Копировать
Редактировать
python scripts/preprocess.py \
  --demographics data/Demographics.csv \
  --response data/Response.csv \
  --questionnaire data/Questionnaire.csv \
  --chemical_labs data/Chemical_Labs.csv \
  --output outputs/Derived/combined_data.csv
Launch Notebooks

Open Jupyter and sequentially run:

01_data_preprocessing.ipynb

02_exploratory_analysis.ipynb

03_model_training.ipynb

04_interpretability_shap.ipynb

05_clustering_dendrogram.ipynb

06_classification_high_ast.ipynb

07_mediation_analysis.ipynb

Automated Scripts (if you prefer non‐interactive execution)

Train Models

bash
Копировать
Редактировать
python scripts/train_models.py \
  --data_path outputs/Derived/combined_data.csv \
  --models_dir outputs/models/ \
  --metrics_path outputs/tables/model_metrics.csv
Interpretability Analysis

bash
Копировать
Редактировать
python scripts/interpretability.py \
  --data_path outputs/Derived/combined_data.csv \
  --models_dir outputs/models/ \
  --out_corr outputs/tables/robust_correlations.csv \
  --out_figures outputs/figures/
Clustering & Dendrogram

bash
Копировать
Редактировать
python scripts/clustering.py \
  --data_path outputs/Derived/combined_data.csv \
  --out_fig outputs/figures/dendrogram.png
Calibration & Classification

bash
Копировать
Редактировать
python scripts/calibration.py \
  --data_path outputs/Derived/combined_data.csv \
  --out_fig_roc outputs/figures/roc_curve_ast.png \
  --out_fig_pr outputs/figures/pr_curve_ast.png \
  --out_cal outputs/figures/calibration_lgbm.png
Mediation Analysis

bash
Копировать
Редактировать
python scripts/mediation.py \
  --data_path outputs/Derived/combined_data.csv \
  --out_csv outputs/tables/mediation_results_all.csv
📈 Key Findings
Stacking_v2 (CatBoost + LightGBM + ExtraTrees → Ridge) yields the highest validation R² (~ 0.92) with minimal overfitting.

Top Predictors (by mean|SHAP| & robust correlations):

Ferritin (LBXFER)

Gamma‐GT (LBXSGTSI)

LDH (LBXSLDSI)

Fasting Glucose (LBXGLU)

Hemoglobin (LBXHGB)

Hierarchical Clustering clearly groups:

Liver/cytolysis enzymes (Ferritin, Gamma‐GT, LDH, ALP)

Metabolic & inflammatory markers (Glucose, LDL, hs-CRP)

Demographics & lifestyle (Age, Sex, Alcohol, Physical Activity)

High‐AST Classification (Gradient Boosting) achieves perfect separation on validation:

ROC AUC = 1.000

AP (PR AUC) = 1.000

Mediation Insights:

Ferritin’s effect on AST is mainly direct (p < 1e-08).

Indirect paths through Gamma-GT, LDH, Glucose, BUN, and Uric Acid are statistically significant, highlighting multi‐pathway regulation.

These results support a multifactorial framework: iron metabolism, hepatocellular injury, and cardiometabolic risk all interplay to shape AST variability in the general population.
