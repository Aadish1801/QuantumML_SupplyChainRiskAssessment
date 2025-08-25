import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import joblib
import os
import time

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

def train_and_evaluate_models():
    """
    Loads the processed data, splits it, trains multiple classical models,
    saves key models, and evaluates their performance.
    """
    # Create results directory if it doesn't exist
    results_dir = 'results/metrics'
    models_dir = 'results/models'
    plots_dir = 'results/plots'
    
    for directory in [results_dir, models_dir, plots_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # 1. Load the processed data
    try:
        X = pd.read_csv('data/processed_features.csv')
        y = pd.read_csv('data/processed_target.csv').iloc[:, 0]
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure 'data/processed_features.csv' and 'data/processed_target.csv' exist.")
        return

    # 2. Split data into training and testing sets
    # Data is already balanced with 400 samples, so no need for SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Data split into training and testing sets.")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    print(f"Class distribution in training set:")
    print(pd.Series(y_train).value_counts().sort_index())
    print(f"Class distribution in testing set:")
    print(pd.Series(y_test).value_counts().sort_index())
    print()

    # Store timing results
    timing_results = []
    
    # 3. Define models to train
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC(probability=True, random_state=42),  # Enable probability for ROC curves
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric='mlogloss')
    }

    # 4. Train, evaluate, and save each model
    for name, model in models.items():
        print(f"--- Evaluating: {name} ---")
        
        # Measure training time
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        
        training_time = end_time - start_time
        print(f"Training time: {training_time:.4f} seconds")

        # Save the model if it's one of the specified ones
        if name in ["Random Forest", "XGBoost"]:
            model_path = os.path.join(models_dir, f'{name.replace(" ", "_")}_model.joblib')
            joblib.dump(model, model_path)
            print(f"Saved trained {name} model to {model_path}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities (for ROC curves)
        if hasattr(model, 'predict_proba'):
            y_probs = model.predict_proba(X_test)
        else:
            y_probs = None
        
        # Evaluate performance
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, target_names=['High Risk', 'Low Risk', 'Moderate Risk'], zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
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
        plot_path = os.path.join(plots_dir, f'{name.replace(" ", "_")}_confusion_matrix.png')
        plt.savefig(plot_path)
        print(f"Saved confusion matrix plot to {plot_path}")
        plt.close()
        
        # Store timing results
        timing_results.append({
            'model_name': name,
            'training_time_seconds': training_time,
            'accuracy': accuracy,
            'f1_score': f1,
            'num_features': X_train.shape[1],
            'num_samples': X_train.shape[0]
        })
    
    # Save timing results
    if timing_results:
        timing_df = pd.DataFrame(timing_results)
        timing_path = os.path.join(results_dir, 'classical_timing_results.csv')
        timing_df.to_csv(timing_path, index=False)
        print(f"Classical model timing results saved to {timing_path}")
        
        # Print summary
        print("\nClassical Model Training Time Summary:")
        print("=" * 50)
        for _, row in timing_df.iterrows():
            print(f"{row['model_name']:<25}: {row['training_time_seconds']:.4f}s (acc: {row['accuracy']:.4f})")

if __name__ == '__main__':
    train_and_evaluate_models()