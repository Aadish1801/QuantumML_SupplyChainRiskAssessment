import pandas as pd
import numpy as np
import time
import os
import pickle
from src.models.vqa_models import VariationalQuantumRiskModels
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

def train_qsvc_models():
    """
    Trains QSVC models across different datasets and hyperparameters, and saves results.
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
    
    # Define the hyperparameter space to explore for QSVC
    param_space = {
        'feature_map_depth': [1, 2, 3],
        'feature_map_entanglement': ['full', 'linear']
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
        print(f"= Starting QSVC Experiments for Dataset: {exp_type.upper()}")
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

        # --- QSVC Hyperparameter Sweep ---
        for feature_map_depth in param_space['feature_map_depth']:
            for entanglement in param_space['feature_map_entanglement']:
                # Define config_str early for use in both try and except blocks
                config_str = f"depth={feature_map_depth}, entanglement={entanglement}"
                print(f"\n--- Testing QSVC with config: {config_str} ---")
                
                # Measure training time
                start_time = time.time()
                try:
                    # Create feature map with current hyperparameters
                    feature_map = ZZFeatureMap(
                        feature_dimension=num_features,
                        reps=feature_map_depth,
                        entanglement=entanglement
                    )
                    
                    # Create quantum kernel with proper sampler
                    from qiskit.primitives import Sampler
                    sampler = Sampler()
                    fidelity = ComputeUncompute(sampler=sampler)
                    quantum_kernel = FidelityQuantumKernel(
                        feature_map=feature_map,
                        fidelity=fidelity
                    )
                    
                    # Create and train QSVC model
                    qsvc = QSVC(quantum_kernel=quantum_kernel)
                    qsvc.fit(X_train_np, y_train_np)
                    
                    end_time = time.time()
                    training_time = end_time - start_time

                    # Evaluate
                    qsvc_score = qsvc.score(X_test_np, y_test_np)
                    
                    print(f"[RESULT] QSVC Accuracy ({exp_type}, {config_str}): {qsvc_score:.4f}")
                    print(f"[TIMING] Training time: {training_time:.4f} seconds")
                    
                    # Save model parameters and circuit information
                    # Following best practices from notebooks/betterment.txt:
                    # - Avoid pickling full Qiskit objects (feature_map, quantum_kernel)
                    # - Store QASM strings or reconstructible info instead for portability
                    # - Include 'error': None for successful runs for consistent DataFrame structure
                    # Note: In newer Qiskit versions, use qasm3.dumps() instead of deprecated .qasm() method
                    import qiskit.qasm3 as qasm3
                    
                    # Convert feature map to QASM3 format
                    feature_map_qasm = qasm3.dumps(feature_map.decompose())
                    
                    model_info = {
                        'model_type': 'QSVC',
                        'experiment_type': exp_type,
                        'num_features': num_features,
                        'feature_map_depth': feature_map_depth,
                        'feature_map_entanglement': entanglement,
                        'feature_map_qasm': feature_map_qasm, # Save as QASM string
                        # 'quantum_kernel': quantum_kernel, # Avoid pickling full kernel object
                        'support_vectors': getattr(qsvc, 'support_vectors_', None),
                        'support_vector_labels': getattr(qsvc, 'y_', None),
                        'n_support': getattr(qsvc, 'n_support_', None),
                        'dual_coef': getattr(qsvc, 'dual_coef_', None),
                        'intercept': getattr(qsvc, 'intercept_', None),
                        'error': None # Indicate success
                    }
                    
                    # Save model info to pickle file
                    model_filename = f'qsvc_model_{exp_type}_depth{feature_map_depth}_{entanglement}.pkl'
                    model_path = os.path.join(models_dir, model_filename)
                    with open(model_path, 'wb') as f:
                        pickle.dump(model_info, f)
                    print(f"[SAVED] Model information saved to {model_path}")
                    
                    # Store timing results - include 'error': None for successful runs
                    timing_results.append({
                        'experiment_type': exp_type,
                        'model_type': 'QSVC',
                        'feature_map_depth': feature_map_depth,
                        'feature_map_entanglement': entanglement,
                        'optimizer': 'N/A',
                        'maxiter': 'N/A',
                        'accuracy': qsvc_score,
                        'training_time_seconds': training_time,
                        'num_features': num_features,
                        'num_samples': len(X_train_np),
                        'error': None # Indicate success
                    })
                    
                except Exception as e:
                    print(f"Error training QSVC for {exp_type} with config {config_str}: {e}")
                    end_time = time.time()
                    training_time = end_time - start_time
                    timing_results.append({
                        'experiment_type': exp_type,
                        'model_type': 'QSVC',
                        'feature_map_depth': feature_map_depth,
                        'feature_map_entanglement': entanglement,
                        'optimizer': 'N/A',
                        'maxiter': 'N/A',
                        'accuracy': 0.0,
                        'training_time_seconds': training_time,
                        'num_features': num_features,
                        'num_samples': len(X_train_np),
                        'error': str(e) # Store the error message
                    })
    
    # Save timing results
    if timing_results:
        timing_df = pd.DataFrame(timing_results)
        timing_path = os.path.join(results_dir, 'qsvc_timing_results.csv')
        timing_df.to_csv(timing_path, index=False)
        print(f"\nQSVC timing results saved to {timing_path}")
        
        # Print summary
        print("\nQSVC Training Time Summary:")
        print("=" * 40)
        for _, row in timing_df.iterrows():
            if 'error' in row and pd.notna(row['error']):
                print(f"{row['experiment_type']} (depth={row['feature_map_depth']}, ent={row['feature_map_entanglement']}): ERROR - {row['error']}")
            else:
                print(f"{row['experiment_type']} (depth={row['feature_map_depth']}, ent={row['feature_map_entanglement']}): "
                      f"{row['training_time_seconds']:.2f}s (acc: {row['accuracy']:.4f})")

if __name__ == '__main__':
    train_qsvc_models()