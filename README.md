# QuantumML_SupplyChainRiskAssessment

This project applies Quantum Machine Learning (QML) techniques to supply chain risk assessment, comparing the performance of classical and quantum models on a curated dataset with comprehensive analysis beyond simple accuracy metrics.

## Project Overview

The goal of this project is to explore how Quantum Machine Learning can be applied to supply chain risk modeling and determine if quantum models can outperform classical approaches, with emphasis on comprehensive analysis including circuit complexity, training dynamics, and resource estimation.

## Key Changes Made

### Data Preprocessing
1. Modified data preprocessing to include previously "leaky" features in the feature set
2. Limited dataset to 400 samples with balanced class distribution
3. Removed SMOTE as data is now balanced through sampling
4. Simplified data splits to only include training/testing (removed validation set)

### Classical Models
1. Removed SMOTE implementation since data is now balanced through sampling
2. Maintained stratified splitting to preserve class balance
3. Added training time measurement for all classical models
4. **NEW**: Added comprehensive visualization capabilities for all models

### Quantum Models
1. Simplified data splits for QML from train/val/test to train/test only
2. Optimized for limited sample size (400 samples)
3. Added training time measurement for quantum models
4. Created enhanced analysis modules for comprehensive evaluation

## Directory Structure

- `src/`: Source code for data preprocessing, classical models, and quantum models
  - `data_preprocessing/`: Scripts for preparing and cleaning the data
  - `models/`: Classical and quantum model implementations
  - `training/`: Scripts for training quantum models
  - `enhanced_analysis/`: Advanced analysis beyond accuracy metrics
- `data/`: Processed datasets and intermediate files
- `results/`: Model outputs, metrics, plots, and paper content
- `notebooks/`: Jupyter notebooks with analysis and visualizations

## Data Pipeline

1. Raw data loading and feature engineering
2. Feature selection using three methods:
   - Principal Component Analysis (PCA)
   - Feature importance from Random Forest
   - SelectKBest with statistical tests
3. Data scaling for quantum angle encoding
4. Dataset splitting into training and testing sets

## Classical Models

Implemented classical models for baseline comparison:
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Random Forest
- Gradient Boosting
- XGBoost

All classical models include:
- Training time measurement
- **NEW**: Advanced visualization capabilities

## Quantum Models

Implemented quantum models:
- Variational Quantum Classifier (VQC)
- Quantum Support Vector Classifier (QSVC) [commented out due to memory issues]

Quantum models include:
- Training time measurement
- Circuit complexity analysis
- Scalability analysis
- Robustness analysis

## Enhanced Analysis Modules

Added comprehensive analysis beyond accuracy:

### Circuit Analysis
- Qubit utilization measurement
- Circuit depth analysis
- Gate composition breakdown
- Parameter count tracking

### Training Dynamics
- Optimizer comparison (SPSA, COBYLA, ADAM)
- Convergence analysis
- Gradient behavior monitoring

### Resource Estimation
- Training time comparison (classical vs quantum)
- Hardware requirements projection
- Noise sensitivity analysis

### Feature Encoding Analysis
- Encoding efficiency comparison
- Quantum feature space visualization
- Class separability measurement

### Theoretical Contributions
- Quantum advantage conditions identification
- Sample complexity analysis
- Generalization studies

### Empirical Analysis
- Robustness studies across multiple runs
- Scalability analysis with qubit count
- Cross-validation performance

## Visualization Capabilities

Created comprehensive visualization tools for:

### Classical Models
- **Enhanced Confusion Matrices**: Improved prediction visualization
- **Feature Importance Plots**: For tree-based models (Random Forest, XGBoost)
- **ROC Curves**: Multiclass receiver operating characteristic analysis
- **Calibration Curves**: Model reliability assessment
- **Model Performance Comparison**: Side-by-side performance metrics

### Quantum Models
- Quantum circuit diagrams
- Training dynamics plots
- Circuit complexity charts
- Scalability studies
- Robustness evaluations

## Requirements

- Python 3.8+
- Qiskit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- XGBoost
- Imbalanced-learn

## Usage

### Basic Pipeline
1. Run data preprocessing:
   ```bash
   python src/data_preprocessing/data_preprocessing.py
   ```

2. Prepare data for QML:
   ```bash
   python src/data_preprocessing/prepare_qml_data_pca.py
   python src/data_preprocessing/prepare_qml_data_feature_selection.py
   python src/data_preprocessing/prepare_qml_data_selectkbest.py
   python src/data_preprocessing/scale_qml_data.py
   python src/data_preprocessing/split_qml_data.py
   ```

3. Train classical models with visualizations:
   ```bash
   python src/models/classical_models.py
   ```

4. Generate advanced model visualizations:
   ```bash
   python src/models/run_visualizations.py
   ```

5. Train quantum models:
   ```bash
   python src/training/train_qml_models.py
   ```

### Enhanced Analysis
6. Run comprehensive enhanced analysis:
   ```bash
   python src/enhanced_analysis/main_driver.py
   ```

7. Generate paper content:
   ```bash
   python src/enhanced_analysis/research_paper_enhancement.py
   ```

## Results

Results are saved in the `results/` directory, including:
- Trained models
- Performance metrics
- Confusion matrices
- Feature importance analyses
- Timing comparisons
- Circuit complexity metrics
- **NEW**: Advanced visualization plots
- Paper content for research publication

## Research Paper Enhancement

The enhanced analysis provides comprehensive content for research paper publication, including:
- Technical analysis beyond accuracy
- Visualization content with circuit diagrams
- Comparative analysis of classical vs quantum approaches
- Resource estimation and hardware requirements
- Theoretical contributions and empirical studies
- **NEW**: Extensive model-specific visualizations for all classical models