from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import SPSA, ADAM, COBYLA
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN # Import SamplerQNN
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from src.models.quantum_circuits import QuantumSupplyChainModels
from qiskit.primitives import Sampler, Estimator

class VariationalQuantumRiskModels:
    def __init__(self, num_qubits=6, ansatz_reps=3, optimizer_name='SPSA', maxiter=100):
        self.q_models = QuantumSupplyChainModels(num_qubits=num_qubits, ansatz_reps=ansatz_reps)
        self.feature_map = self.q_models.create_feature_map()
        self.ansatz = self.q_models.create_ansatz()
        self.sampler = Sampler() # This is fine for SamplerQNN
        self.estimator = Estimator() # This is fine for EstimatorQNN
        
        # Select the optimizer based on the input name
        if optimizer_name.upper() == 'ADAM':
            self.optimizer = ADAM(maxiter=maxiter)
        elif optimizer_name.upper() == 'COBYLA':
            self.optimizer = COBYLA(maxiter=maxiter)
        else: # Default to SPSA
            self.optimizer = SPSA(maxiter=maxiter)

    def create_qnn(self):
        # Combine the feature map and ansatz into a single circuit
        qc = QuantumCircuit(self.q_models.num_qubits)
        qc.compose(self.feature_map, inplace=True)
        qc.compose(self.ansatz, inplace=True)
        # Add measurements to all qubits for SamplerQNN
        qc.measure_all() 

        # Use SamplerQNN for classification tasks, it's generally more suitable
        # The output shape for SamplerQNN is handled differently, often based on the number of qubits measured
        # and the interpret function if needed. For standard classification with NeuralNetworkClassifier,
        # it often works automatically based on the y_train labels.
        qnn = SamplerQNN(
            circuit=qc, 
            input_params=self.feature_map.parameters, 
            weight_params=self.ansatz.parameters,
            sampler=self.sampler
            # output_shape is typically not needed here for standard use with NeuralNetworkClassifier
            # The classifier infers the number of classes from y_train
        )
        return qnn

    def create_vqc(self):
        vqc = VQC(
        feature_map=self.feature_map,
        ansatz=self.ansatz,
        optimizer=self.optimizer
    )
        return vqc


    def create_qsvc(self):
        fidelity = ComputeUncompute(sampler=self.sampler)
        quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=fidelity
        )
        qsvc = QSVC(quantum_kernel=quantum_kernel)
        return qsvc
