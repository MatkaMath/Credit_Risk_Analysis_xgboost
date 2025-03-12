import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

def generate_plots(X_train, X_train_pca, preprocessor, pca, cat_cols, y_true=None, y_scores=None):
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    
    # Compute correlation matrix for numerical features
    num_corr = X_train[num_cols].corr()
    
    # Compute correlation matrix for one-hot encoded categorical features
    cat_encoded_columns = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
    X_train_cat_df = pd.DataFrame(
        preprocessor.transform(X_train)[:, len(num_cols):],
        columns=cat_encoded_columns
    )
    cat_corr = X_train_cat_df.corr()
    
    # Remove highly correlated categorical features (above 0.9 threshold)
    to_drop = set()
    for i in range(len(cat_corr.columns)):
        for j in range(i + 1, len(cat_corr.columns)):
            if abs(cat_corr.iloc[i, j]) > 0.9:
                to_drop.add(cat_corr.columns[j])  # Drop second feature in the pair
    
    X_train_cat_df = X_train_cat_df.drop(columns=to_drop, errors='ignore')
    
    # Plot both heatmaps using subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(num_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0])
    axes[0].set_title("Numerical Feature Correlation Matrix")
    
    if not X_train_cat_df.empty:
        sns.heatmap(X_train_cat_df.corr(), annot=False, cmap='coolwarm', center=0, ax=axes[1])
        axes[1].set_title("Categorical Feature Correlation Matrix (After Removal)")
    else:
        print("No categorical features left after removing high correlations.")
        axes[1].set_visible(False)
    
    plt.savefig("results/feature_correlation_matrix.png")
    
    # PCA Explained Variance
    plt.figure(figsize=(10, 5))
    components = [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))]
    plt.barh(components[::-1], pca.explained_variance_ratio_[::-1], color='b')
    plt.xlabel('Explained Variance Ratio')
    plt.ylabel('Principal Components')
    plt.title('PCA Component Explained Variance')
    plt.savefig("results/pca_explained_variance.png")

    # PCA Cumulative Explained Variance
    plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.grid()
    plt.savefig("results/pca_cumulative_variance.png")
    
    # PCA Loadings Heatmap
    feature_names = list(num_cols) + list(cat_encoded_columns)
    plt.figure(figsize=(14, 6))
    pca_loadings = pd.DataFrame(pca.components_.T, index=feature_names, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    sns.heatmap(pca_loadings, cmap="coolwarm", center=0, annot=False)
    plt.ylabel("Original Features")
    plt.xlabel("Principal Components")
    plt.title("PCA Loadings Heatmap")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.savefig("results/pca_loadings_heatmap.png")
    
    # ROC Curve
    if y_true is not None and y_scores is not None:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        
        plt.savefig("results/roc_curve.png")
