from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import SPSA, ADAM, COBYLA
from qiskit_machine_learning.algorithms import VQC, QSVC
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN # Import SamplerQNN
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_algorithms.state_fidelities import ComputeUncompute
from src.models.quantum_circuits import QuantumSupplyChainModels
from qiskit.primitives import BackendSampler
from qiskit_aer import AerSimulator

class VariationalQuantumRiskModels:
    def __init__(self, num_qubits=6, ansatz_reps=3, feature_map_reps=2, feature_map_entanglement='linear', optimizer_name='SPSA', maxiter=100):
        self.q_models = QuantumSupplyChainModels(
            num_qubits=num_qubits, 
            ansatz_reps=ansatz_reps,
            feature_map_reps=feature_map_reps,
            feature_map_entanglement=feature_map_entanglement
        )
        self.feature_map = self.q_models.create_feature_map()
        self.ansatz = self.q_models.create_ansatz()
        
        # Use explicit AerSimulator for better control and performance
        backend = AerSimulator(method='statevector', shots=1024)
        self.sampler = BackendSampler(backend=backend, options={"shots": 1024})
        self.sampler.set_options(seed=42)  # For reproducibility
        
        # Select the optimizer based on the input name
        if optimizer_name.upper() == 'ADAM':
            self.optimizer = ADAM(maxiter=maxiter)
        elif optimizer_name.upper() == 'COBYLA':
            self.optimizer = COBYLA(maxiter=maxiter)
        else: # Default to SPSA
            self.optimizer = SPSA(maxiter=maxiter)

    def create_qnn(self, num_classes=3):
        # Combine the feature map and ansatz into a single circuit
        qc = QuantumCircuit(self.q_models.num_qubits)
        qc.compose(self.feature_map, inplace=True)
        qc.compose(self.ansatz, inplace=True)
        # Add measurements to all qubits for SamplerQNN
        qc.measure_all() 

        # Define an interpret function to map bitstrings to classes
        # This simple approach uses modulo to distribute outcomes among classes
        def interpret_class(x):
            # x is the integer representation of the bitstring
            # Map to classes 0, 1, 2 using modulo
            return x % num_classes

        # Use SamplerQNN with proper interpret function and output shape
        qnn = SamplerQNN(
            circuit=qc, 
            input_params=self.feature_map.parameters, 
            weight_params=self.ansatz.parameters,
            sampler=self.sampler,
            interpret=interpret_class,
            output_shape=(num_classes,)  # Set output shape to match number of classes
        )
        return qnn

    def create_vqc(self):
        vqc = VQC(
        feature_map=self.feature_map,
        ansatz=self.ansatz,
        optimizer=self.optimizer
    )
        return vqc


    def create_qsvc(self, C=1.0):
        fidelity = ComputeUncompute(sampler=self.sampler)
        quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map,
            fidelity=fidelity
        )
        qsvc = QSVC(quantum_kernel=quantum_kernel, C=C)
        return qsvc
