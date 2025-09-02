from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

class QuantumSupplyChainModels:
    def __init__(self, num_qubits=6, ansatz_reps=3, feature_map_reps=2, feature_map_entanglement='linear'):
        self.num_qubits = num_qubits
        self.ansatz_reps = ansatz_reps
        self.feature_map_reps = feature_map_reps
        self.feature_map_entanglement = feature_map_entanglement

    def create_feature_map(self):
        # ZZ Feature Map for supply chain features
        feature_map = ZZFeatureMap(
            feature_dimension=self.num_qubits,
            reps=self.feature_map_reps,
            entanglement=self.feature_map_entanglement
        )
        return feature_map

    def create_ansatz(self):
        # Parameterized quantum circuit
        ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=self.ansatz_reps,
            entanglement='linear'  # Using linear to keep circuits manageable
        )
        return ansatz
