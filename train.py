import xgboost as xgb
from sklearn.decomposition import PCA
import numpy as np
import joblib  # For saving the model & PCA
from preprocess import load_and_preprocess_data

def train_model(X_train, X_test, y_train, y_test):
    # Load preprocessor
    _, _, _, _, preprocessor, _ = load_and_preprocess_data()

    # Transform data
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Apply PCA
    pca = PCA(n_components=12)
    X_train_pca = pca.fit_transform(X_train_transformed)
    X_test_pca = pca.transform(X_test_transformed)

    # Train model
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    model.fit(X_train_pca, y_train)

    # Save model and PCA transformer
    joblib.dump(model, "models/xgboost_model.pkl")
    joblib.dump(pca, "models/pca_transformer.pkl")

    # Save test data for evaluation
    np.save("models/X_test_pca.npy", X_test_pca)
    np.save("models/y_test.npy", y_test)

    return model, X_train_pca, X_test_pca, pca  # ✅ Returning necessary values

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor, _ = load_and_preprocess_data()
    train_model(X_train, X_test, y_train, y_test)
