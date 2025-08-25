"""
Comprehensive Model Visualization Script

This script loads all trained models and generates visualizations for inclusion in the research paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

class ModelVisualizations:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        self.plots_dir = os.path.join(results_dir, 'plots')
        self.models_dir = os.path.join(results_dir, 'models')
        self.metrics_dir = os.path.join(results_dir, 'metrics')
        
        # Create directories if they don't exist
        for directory in [self.plots_dir, self.models_dir, self.metrics_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def load_data(self):
        """
        Load the processed data for visualization.
        """
        try:
            X = pd.read_csv('data/processed_features.csv')
            y = pd.read_csv('data/processed_target.csv').iloc[:, 0]
            
            # Split data into training and testing sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            return X_train, X_test, y_train, y_test
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run data preprocessing first.")
            return None, None, None, None
    
    def load_trained_models(self):
        """
        Load all trained models from the models directory.
        """
        models = {}
        
        # Define model paths
        model_paths = {
            'Random_Forest': os.path.join(self.models_dir, 'Random_Forest_model.joblib'),
            'XGBoost': os.path.join(self.models_dir, 'XGBoost_model.joblib')
        }
        
        # Try to load each model
        for model_name, model_path in model_paths.items():
            try:
                model = joblib.load(model_path)
                models[model_name] = model
                print(f"Loaded {model_name} model")
            except FileNotFoundError:
                print(f"Warning: {model_name} model not found at {model_path}")
            except Exception as e:
                print(f"Error loading {model_name} model: {e}")
        
        return models
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name, class_names=['High Risk', 'Low Risk', 'Moderate Risk']):
        """
        Plot and save confusion matrix for a model.
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, 
                    yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, f'{model_name.replace(" ", "_")}_confusion_matrix_enhanced.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved enhanced confusion matrix for {model_name} to {plot_path}")
        return cm
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=15):
        """
        Plot feature importance for tree-based models.
        """
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            print(f"Model {model_name} does not have feature importances")
            return
        
        # Create DataFrame for easier handling
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Select top N features
        top_features = feature_imp_df.head(top_n)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        bars = plt.barh(range(len(top_features)), top_features['importance'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances - {model_name}')
        plt.gca().invert_yaxis()  # Highest importance at top
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(top_features.iterrows()):
            plt.text(row['importance'] + 0.001, i, f'{row["importance"]:.3f}', 
                     va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, f'{model_name.replace(" ", "_")}_feature_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved feature importance plot for {model_name} to {plot_path}")
        
        # Save feature importance data
        csv_path = os.path.join(self.metrics_dir, f'{model_name.replace(" ", "_")}_feature_importance.csv')
        feature_imp_df.to_csv(csv_path, index=False)
        print(f"Saved feature importance data for {model_name} to {csv_path}")
    
    def plot_roc_curves(self, y_true, y_scores, model_name, class_names=['High Risk', 'Low Risk', 'Moderate Risk']):
        """
        Plot ROC curves for multiclass classification.
        """
        # Binarize the output labels for multiclass ROC
        y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
        n_classes = y_true_bin.shape[1]
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        
        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average ROC curve (AUC = {roc_auc["micro"]:.2f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        # Plot ROC curve for each class
        colors = ['aqua', 'darkorange', 'cornflowerblue']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of {class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multiclass ROC Curves - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, f'{model_name.replace(" ", "_")}_roc_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC curves for {model_name} to {plot_path}")
        
        # Save ROC data
        roc_data = []
        for i in range(n_classes):
            roc_data.append({
                'class': class_names[i],
                'fpr': list(fpr[i]),
                'tpr': list(tpr[i]),
                'auc': roc_auc[i]
            })
        roc_data.append({
            'class': 'micro_average',
            'fpr': list(fpr["micro"]),
            'tpr': list(tpr["micro"]),
            'auc': roc_auc["micro"]
        })
        
        csv_path = os.path.join(self.metrics_dir, f'{model_name.replace(" ", "_")}_roc_data.csv')
        pd.DataFrame(roc_data).to_csv(csv_path, index=False)
        print(f"Saved ROC data for {model_name} to {csv_path}")
    
    def plot_calibration_curves(self, y_true, y_probs, model_name, class_names=['High Risk', 'Low Risk', 'Moderate Risk']):
        """
        Plot calibration curves for each class.
        """
        plt.figure(figsize=(10, 8))
        
        # Plot perfectly calibrated line
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        # Plot calibration curve for each class
        for i, class_name in enumerate(class_names):
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true == i, y_probs[:, i], n_bins=10
            )
            
            plt.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label=f"{class_name}")
        
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted value")
        plt.ylim([-0.05, 1.05])
        plt.legend(loc="lower right")
        plt.title(f'Calibration plots - {model_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, f'{model_name.replace(" ", "_")}_calibration.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved calibration curves for {model_name} to {plot_path}")
    
    def plot_model_comparison(self, model_metrics):
        """
        Plot comparison of models across different metrics.
        """
        # Convert to DataFrame
        metrics_df = pd.DataFrame(model_metrics)
        
        # Set up the matplotlib figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Plot 1: Accuracy Comparison
        axes[0, 0].bar(metrics_df['model'], metrics_df['accuracy'], color='skyblue')
        axes[0, 0].set_title('Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Precision Comparison
        axes[0, 1].bar(metrics_df['model'], metrics_df['precision'], color='lightcoral')
        axes[0, 1].set_title('Precision Comparison')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Recall Comparison
        axes[1, 0].bar(metrics_df['model'], metrics_df['recall'], color='lightgreen')
        axes[1, 0].set_title('Recall Comparison')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: F1-Score Comparison
        axes[1, 1].bar(metrics_df['model'], metrics_df['f1_score'], color='gold')
        axes[1, 1].set_title('F1-Score Comparison')
        axes[1, 1].set_ylabel('F1-Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'model_performance_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved model comparison plot to {plot_path}")
    
    def generate_all_visualizations(self):
        """
        Generate all visualizations for trained models.
        """
        print("Generating comprehensive model visualizations...")
        print("=" * 55)
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        if X_train is None:
            return
        
        # Load trained models
        models = self.load_trained_models()
        if not models:
            print("No trained models found. Please train models first.")
            return
        
        # Get feature names
        feature_names = X_train.columns.tolist()
        
        # Store model metrics for comparison
        model_metrics = []
        
        # Generate visualizations for each model
        for model_name, model in models.items():
            print(f"\nGenerating visualizations for {model_name}...")
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities (if available)
            if hasattr(model, 'predict_proba'):
                y_probs = model.predict_proba(X_test)
            else:
                y_probs = None
            
            # 1. Confusion Matrix
            cm = self.plot_confusion_matrix(y_test, y_pred, model_name)
            
            # 2. Feature Importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                self.plot_feature_importance(model, feature_names, model_name)
            
            # 3. ROC Curves (if probabilities are available)
            if y_probs is not None:
                self.plot_roc_curves(y_test, y_probs, model_name)
                self.plot_calibration_curves(y_test, y_probs, model_name)
            
            # 4. Collect metrics for comparison
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            model_metrics.append({
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
        
        # 5. Model Comparison Plot
        if model_metrics:
            self.plot_model_comparison(model_metrics)
        
        print(f"\nAll visualizations saved to {self.plots_dir}")
        print("All metrics saved to {self.metrics_dir}")
        print("\nVisualization generation complete!")

def main():
    """
    Main function to run the model visualization script.
    """
    print("Comprehensive Model Visualization Script")
    print("=" * 45)
    
    # Initialize visualizer
    visualizer = ModelVisualizations()
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()