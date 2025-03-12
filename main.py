from preprocess import load_and_preprocess_data
from train import train_model
from predict import evaluate_model
from visualizations import generate_plots

def main():
    # Load preprocessed data
    X_train, X_test, y_train, y_test, preprocessor, cat_cols = load_and_preprocess_data()

    # Train the model
    model, X_train_pca, X_test_pca, pca = train_model(X_train, X_test, y_train, y_test)

    # Evaluate the model
    y_test, y_test_scores = evaluate_model(model, X_test_pca, y_test)

    # Generate plots
    generate_plots(X_train, X_train_pca, preprocessor, pca, cat_cols, y_test, y_test_scores)

if __name__ == "__main__":
    main()
