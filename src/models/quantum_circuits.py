from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

class QuantumSupplyChainModels:
    def __init__(self, num_qubits=6, ansatz_reps=3):
        self.num_qubits = num_qubits
        self.ansatz_reps = ansatz_reps

    def create_feature_map(self):
        # ZZ Feature Map for supply chain features
        feature_map = ZZFeatureMap(
            feature_dimension=self.num_qubits,
            reps=2,
            entanglement='linear'
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
