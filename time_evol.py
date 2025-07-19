import numpy as np
from src.circuit import Circuit

if __name__ == "__main__":
    Ls = [4, 6, 8, 10]
    n_circuits = 60
    n_prob = 21
    steps = 60
    p = np.linspace(0, 0.4, n_prob)  # Probability of applying a unitary operator
    circuits = np.zeros((len(Ls), n_prob), dtype=object)
    entropies = np.zeros((len(Ls), n_circuits, steps + 1, n_prob))
    for idx, l in enumerate(Ls):
        for j, prob in enumerate(p):
            for k in range(n_circuits):
                circuit = Circuit(l, prob)
                entropies[idx, k, 0, j] = circuit.state.entanglement_entropy()
                for i in range(1, steps + 1):
                    circuit.full_circuit_evolution(1)
                    entropies[idx, k, i, j] = circuit.state.entanglement_entropy()
    data_list = []
    for l_idx, L in enumerate(Ls):
        for j in range(n_circuits):
            for i in range(steps + 1):
                for k in range(n_prob):
                    data_list.append(
                        [
                            L,  # System size
                            i,  # Time step
                            j,  # Circuit number
                            p[k],  # Probability
                            entropies[l_idx, j, i, k],  # Entropy value
                        ]
                    )

    # Convert to numpy array
    data_array = np.array(data_list)

    # Save with header
    header = "L,timestep,circuit_num,probability,entropy"
    np.savetxt(
        "entropies_structured_pbc.csv", data_array, fmt="%d,%d,%d,%.6f,%.6f", delimiter=",", header=header, comments=""
    )
