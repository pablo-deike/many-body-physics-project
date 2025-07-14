import numpy as np
from src.state import State
from src.circuit import Circuit
import matplotlib.pyplot as plt
from numpy.typing import NDArray

if __name__ == "__main__":
    L = 6
    # bell_state = np.array([1, 0, 0, 1]) / np.sqrt(2)  # Bell state |Φ+⟩ = (|00⟩ + |11⟩) / √2

    # circuit = Circuit(L=L, p=0.2)
    # circuit.full_circuit_evolution(t=10)

    # print("Final state after applying the circuit:")
    # print("norm of the final state:")
    # print("Entanglement entropy of the final state:")
    # print(circuit.state.entanglement_entropy())

    circuits: list[Circuit] = []
    n_circuits = 10
    n_prob = 4
    p = np.linspace(0, 0.5, n_prob)  # Probability of applying a unitary operator
    for prob in p:
        for _ in range(n_circuits):
            circuits.append(Circuit(L, prob))
    # Perform full circuit evolution
    circuits = np.array(circuits)
    steps = 100
    entropies = np.zeros((n_circuits, steps + 1, n_prob))
    for l, circuit in enumerate(circuits):
        k = l // n_circuits
        j = l % n_circuits
        entropies[j, 0, k] = circuit.state.entanglement_entropy()[L // 2]
        for i in range(1, steps + 1):
            circuit.full_circuit_evolution(1)
            entropies[j, i, k] = circuit.state.entanglement_entropy()[L // 2]

    for k in range(n_prob):
        plt.plot(np.average(entropies[:, :, k], axis=0) / L, label=f"p={p[k]:.2f}")
    plt.xlabel("Time step")
    plt.ylabel("Entanglement entropy")
    plt.title("Entanglement entropy over time")
    plt.legend()
    plt.grid()
    plt.show()
