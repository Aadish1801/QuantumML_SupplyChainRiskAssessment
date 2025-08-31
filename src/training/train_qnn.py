import pandas as pd
import numpy as np
import time
import os
import pickle
from src.models.vqa_models import VariationalQuantumRiskModels
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import SPSA, ADAM, COBYLA
from sklearn.metrics import accuracy_score

def train_qnn_models():
    """
    Trains QNN models across different datasets and hyperparameters, and saves results.
    """
    # Create results directory if it doesn't exist
    results_dir = 'results/metrics'
    models_dir = 'results/models'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Create timing results file
    timing_results = []
    
    # Define the hyperparameter space to explore for QNN
    param_space = {
        'ansatz_reps': [2, 3],
        'optimizers': {
            'SPSA': [100, 150],
            'ADAM': [60, 80],
            'COBYLA': [150, 200]
        }
    }
    
    # Define the different experimental datasets
    experiment_types = ['pca', 'feature_selection', 'selectkbest']

    # Load the common target variable files
    try:
        y_train = pd.read_csv('data/y_train.csv').iloc[:, 0]
        y_test = pd.read_csv('data/y_test.csv').iloc[:, 0]
    except FileNotFoundError as e:
        print(f"FATAL: Could not load target files: {e}")
        return

    # --- Main Loop for Experiments ---
    for exp_type in experiment_types:
        print(f"\n{'='*60}")
        print(f"= Starting QNN Experiments for Dataset: {exp_type.upper()}")
        print(f"{'='*60}")

        # Load the feature data for the current experiment type
        try:
            X_train = pd.read_csv(f'data/X_train_{exp_type}.csv')
            X_test = pd.read_csv(f'data/X_test_{exp_type}.csv')
        except FileNotFoundError:
            print(f"Warning: Data files for '{exp_type}' not found. Skipping.")
            continue

        num_features = X_train.shape[1]
        X_train_np, y_train_np = X_train.to_numpy(), y_train.to_numpy()
        X_test_np, y_test_np = X_test.to_numpy(), y_test.to_numpy()

        # --- QNN Hyperparameter Sweep ---
        for ansatz_reps in param_space['ansatz_reps']:
            for optimizer_name, max_iters in param_space['optimizers'].items():
                for maxiter in max_iters:
                    # Define config_str early for use in both try and except blocks
                    config_str = f"reps={ansatz_reps}, opt={optimizer_name}, iter={maxiter}"
                    print(f"\n--- Testing QNN with config: {config_str} ---")
                    
                    # Select the optimizer based on the input name
                    if optimizer_name.upper() == 'ADAM':
                        optimizer = ADAM(maxiter=maxiter)
                    elif optimizer_name.upper() == 'COBYLA':
                        optimizer = COBYLA(maxiter=maxiter)
                    else:  # Default to SPSA
                        optimizer = SPSA(maxiter=maxiter)
                    
                    # Initialize training time
                    training_time = 0.0
                    # Measure training time
                    start_time = time.time()
                    try:
                        # Configure and initialize the QNN model
                        qnn_models = VariationalQuantumRiskModels(
                            num_qubits=num_features,
                            ansatz_reps=ansatz_reps
                        )
                        # Get number of classes from the dataset
                        num_classes = len(np.unique(y_train_np))
                        qnn = qnn_models.create_qnn(num_classes=num_classes)
                        
                        # Create classifier with QNN and proper loss function for classification
                        classifier = NeuralNetworkClassifier(
                            neural_network=qnn,
                            optimizer=optimizer,
                            loss='cross_entropy',  # Use cross-entropy loss for classification
                            one_hot=True  # Assuming labels need to be one-hot encoded
                        )
                        
                        classifier.fit(X_train_np, y_train_np)
                        end_time = time.time()
                        
                        training_time = end_time - start_time

                        # Evaluate
                        y_pred = classifier.predict(X_test_np)
                        qnn_score = accuracy_score(y_test_np, y_pred)
                        
                        print(f"[RESULT] QNN Accuracy ({exp_type}, {config_str}): {qnn_score:.4f}")
                        print(f"[TIMING] Training time: {training_time:.4f} seconds")
                        
                        # Extract final loss safely
                        final_loss = None
                        classifier_fit_result = getattr(classifier, 'fit_result', None)
                        if classifier_fit_result is not None:
                            # Qiskit's OptimizerResult typically has a 'fun' attribute for the final function value
                            # or 'loss' depending on the version or specific optimizer used.
                            # Check for common attribute names.
                            if hasattr(classifier_fit_result, 'fun'):
                                final_loss = classifier_fit_result.fun
                            elif hasattr(classifier_fit_result, 'loss'):
                                final_loss = classifier_fit_result.loss

                        # Save model parameters and circuit information
                        # Following best practices from notebooks/betterment.txt:
                        # - Avoid pickling full Qiskit objects (feature_map, ansatz)
                        # - Store QASM strings instead for portability
                        # - Optimizer name and maxiter are already stored
                        # - Include 'error': None for successful runs for consistent DataFrame structure
                        # Note: In newer Qiskit versions, use qasm3.dumps() instead of deprecated .qasm() method
                        import qiskit.qasm3 as qasm3
                        
                        # Convert feature map and ansatz to QASM3 format
                        feature_map_qasm = qasm3.dumps(qnn_models.feature_map.decompose())
                        ansatz_qasm = qasm3.dumps(qnn_models.ansatz.decompose())
                        
                        model_info = {
                            'model_type': 'QNN',
                            'experiment_type': exp_type,
                            'num_features': num_features,
                            'ansatz_reps': ansatz_reps,
                            'optimizer': optimizer_name,
                            'maxiter': maxiter,
                            'feature_map_qasm': feature_map_qasm, # Save as QASM string
                            'ansatz_qasm': ansatz_qasm,       # Save as QASM string
                            'weights': getattr(classifier, 'weights', None),
                            'optimizer_obj': f"{optimizer_name}(maxiter={maxiter})", # Store optimizer info as string, not object
                            'final_loss': final_loss, # Use the safely extracted final_loss
                            'error': None # Indicate success
                        }
                        
                        # Save model info to pickle file
                        model_filename = f'qnn_model_{exp_type}_reps{ansatz_reps}_{optimizer_name}_{maxiter}.pkl'
                        model_path = os.path.join(models_dir, model_filename)
                        with open(model_path, 'wb') as f:
                            pickle.dump(model_info, f)
                        print(f"[SAVED] Model information saved to {model_path}")
                        
                        # Store timing results - include 'error': None for successful runs
                        timing_results.append({
                            'experiment_type': exp_type,
                            'model_type': 'QNN',
                            'ansatz_reps': ansatz_reps,
                            'optimizer': optimizer_name,
                            'maxiter': maxiter,
                            'accuracy': qnn_score,
                            'training_time_seconds': training_time,
                            'num_features': num_features,
                            'num_samples': len(X_train_np),
                            'error': None # Indicate success
                        })
                        
                    except Exception as e:
                        end_time = time.time()
                        training_time = end_time - start_time # Calculate time even if failed
                        # config_str is now guaranteed to be defined
                        print(f"Error training QNN for {exp_type} with config {config_str}: {e}")
                        timing_results.append({
                            'experiment_type': exp_type,
                            'model_type': 'QNN',
                            'ansatz_reps': ansatz_reps,
                            'optimizer': optimizer_name,
                            'maxiter': maxiter,
                            'accuracy': 0.0,
                            'training_time_seconds': training_time,
                            'num_features': num_features,
                            'num_samples': len(X_train_np),
                            'error': str(e) # Store the error message
                        })
    
    # Save timing results
    if timing_results:
        timing_df = pd.DataFrame(timing_results)
        timing_path = os.path.join(results_dir, 'qnn_timing_results.csv')
        timing_df.to_csv(timing_path, index=False)
        print(f"\nQNN timing results saved to {timing_path}")
        
        # Print summary
        print("\nQNN Training Time Summary:")
        print("=" * 40)
        for _, row in timing_df.iterrows():
            if 'error' in row and pd.notna(row['error']):
                print(f"{row['experiment_type']} ({row['optimizer']}, reps={row['ansatz_reps']}): ERROR - {row['error']}")
            else:
                print(f"{row['experiment_type']} ({row['optimizer']}, reps={row['ansatz_reps']}): "
                      f"{row['training_time_seconds']:.2f}s (acc: {row['accuracy']:.4f})")

if __name__ == '__main__':
    train_qnn_models()