# Credit Risk Analysis with PCA and XGBoost

## 📌 Overview
This project performs **Credit Risk Analysis** using the **UCI Credit Approval dataset**.  
It aims to **predict whether a loan applicant is risky or creditworthy** based on financial and demographic data.

### 🔹 Key Steps:
- **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling numerical features.
- **Feature Engineering**: Creating new features and removing highly correlated ones.
- **Dimensionality Reduction**: Applying **Principal Component Analysis (PCA)** to reduce dimensionality.
- **Machine Learning Model**: Training an **XGBoost classifier** for prediction.
- **Model Evaluation**: Using **ROC-AUC score**, **classification metrics**, and visualizations.
- **Data Visualization**: Generating key plots to better understand data relationships and model performance.

---

## 📊 Dataset
The dataset is sourced from the **UCI Machine Learning Repository**:  
[📂 Credit Approval Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data)  

- **Rows**: ~690 samples  
- **Features**: 15 (Categorical + Numerical)  
- **Target Variable (`TARGET`)**:  
  - `1` → Credit Approved  
  - `0` → Credit Denied  

---

## 📈 Visualizations
The project includes several key visualizations for analysis:

1. **Numerical & Categorical Feature Correlation Matrix** (`correlation_matrix.png`)
   - Displays correlations between numerical features and encoded categorical features.
   
2. **Cumulative Explained Variance by PCA Components** (`pca_explained_variance.png`)
   - Shows how much variance is retained with different numbers of PCA components.
   
3. **PCA Component Explained Variance** (`pca_component_variance.png`)
   - Highlights the importance of each principal component in explaining variance.
   
4. **PCA Loadings Heatmap** (`pca_loadings_heatmap.png`)
   - Illustrates the contributions of original features to the principal components.

5. **ROC Curve** (`roc_curve.png`)
   - Evaluates model performance by plotting the True Positive Rate (TPR) vs. False Positive Rate (FPR).

---

## 🚀 Running the Project
### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Pipeline
```bash
python main.py
```

### 3️⃣ Check Outputs
- Trained models will be saved in the `models/` directory.
- Evaluation results and visualizations will be stored in the `results/` directory.

---

## 📂 Project Structure
```
📦 Credit_Risk_Analysis
├── 📄 main.py             # Runs preprocessing, training, and evaluation
├── 📄 preprocess.py       # Data loading and preprocessing
├── 📄 train.py            # Model training using PCA and XGBoost
├── 📄 predict.py          # Model evaluation and predictions
├── 📄 visualizations.py   # Generates data visualizations
├── 📄 requirements.txt    # Dependencies
├── 📂 models/             # Saved trained models
└── 📂 results/            # Evaluation metrics & visualizations
```

---

## 📊 Results & Evaluation
After training, the model's performance is evaluated using:
- **ROC-AUC Score** to measure classification quality.
- **Classification Report** with precision, recall, and F1-score.
- **Visualizations** to interpret feature relationships and PCA effects.

---

## 🔗 References
- [UCI Machine Learning Repository - Credit Approval Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/)

---

🚀 **Ready to analyze credit risk? Run the pipeline and explore the results!**
