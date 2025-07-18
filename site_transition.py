import numpy as np
from src.circuit import Circuit

if __name__ == "__main__":
    L = 10
    n_circuits = 60
    n_prob = 21
    steps = 60
    p = np.linspace(0, 0.4, n_prob)  # Probability of applying a unitary operator
    entropies = np.zeros((n_circuits, n_prob, L //2))
    for j, prob in enumerate(p):
        for k in range(n_circuits):
            circuit = Circuit(L, prob)
            for i in range(L // 2):
                circuit.full_circuit_evolution(steps)
                entropies[k, j, i] = circuit.state.entanglement_entropy(i+1)
    data_list = []
    for j in range(n_circuits):
        for k in range(n_prob):
            for i in range(L // 2):
                data_list.append([
                j,                           # Circuit number
                p[k],                        # Probability
                entropies[j, k, i]    # Entropy value
            ])

    # Convert to numpy array
    data_array = np.array(data_list)

    # Save with header
    header = "circuit_num,probability,entropy"
    np.savetxt(f"entropies_structured_pbc_sites_L_{L}.csv", data_array, fmt='%d,%d,%d,%.6f,%.6f', 
            delimiter=',', header=header, comments='')
