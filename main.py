import numpy as np
from src.state import State
from src.circuit import Circuit
import matplotlib.pyplot as plt
from numpy.typing import NDArray

if __name__ == "__main__":
    L = 4  # Number of sites
    # bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2

    circuit = Circuit(L=L, p=0.5, initial_state=State(L))
    circuit.state.create_all_measurement_sigz()
    circuit.full_circuit_evolution(t=10)

    print("Final state after applying the circuit:")
    print(circuit.state.array)
    print("norm of the final state:")
    print(np.linalg.norm(circuit.state.array))
    print("Entanglement entropy of the final state:")
    print(circuit.state.entanglement_entropy())

    # for k in range(n_prob):
    #     plt.plot(np.average(entropies[:, :, k], axis=0) / L, label=f"p={p[k]:.2f}")
    # plt.xlabel("Time step")
    # plt.ylabel("Entanglement entropy")
    # plt.title("Entanglement entropy over time")
    # plt.legend()
    # plt.grid()
    # plt.show()
