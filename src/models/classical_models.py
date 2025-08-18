
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

def train_and_evaluate_models():
    """
    Loads the processed data, splits it, trains multiple classical models,
    and evaluates their performance.
    """
    # 1. Load the processed data
    try:
        X = pd.read_csv('data/processed_features.csv')
        y = pd.read_csv('data/processed_target.csv').iloc[:, 0] # Read the first column as a Series
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'data/processed_features.csv' and 'data/processed_target.csv' are in the same directory.")
        return

    # 2. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print(f"Data split into training and testing sets.")
    print(f"Original training set shape: {X_train.shape}")
    print(f"Resampled training set shape: {X_train_resampled.shape}")
    print(f"Testing set shape: {X_test.shape}\n")

    # 3. Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss', use_label_encoder=False)
    }


    # 4. Train and evaluate each model
    for name, model in models.items():
        print(f"--- Evaluating: {name} ---")
        
        # Train the model
        model.fit(X_train_resampled, y_train_resampled)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate performance
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=['High Risk', 'Low Risk', 'Moderate Risk'], zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Weighted F1 Score: {f1:.4f}")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['High', 'Low', 'Moderate'], 
                    yticklabels=['High', 'Low', 'Moderate'])
        plt.title(f'Confusion Matrix for {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        # Saving the plot to a file
        plt.savefig(f'results/metrics/{name.replace(" ", "_")}_confusion_matrix.png')
        print(f"Saved confusion matrix plot to KNN_confusion_matrix.png\n")


if __name__ == '__main__':
    train_and_evaluate_models()
