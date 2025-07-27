import numpy as np
from numpy.typing import ArrayLike


class State:
    """Quantum state representation for many-body systems."""
    
    def __init__(self, L: int, array: ArrayLike | None = None) -> None:
        """Initialize a quantum state.
        
        Args:
            L: Number of sites in the system
            array: Optional state vector (defaults to all-zero state)
        """
        self.dim = 2**L
        self.L = L
        if array is None:
            self.init_all_zeros()
        else:
            if len(array) != self.dim:
                raise ValueError(f"State must have length {self.dim}, got {len(array)}")
            self.array = np.array(array, dtype=np.complex128)
        self.array = self.array.flatten()
        self.array = np.reshape(self.array, [2] * L)

    def init_all_zeros(self) -> None:
        """Initialize the state to the all-zero state |00...0⟩."""
        self.array = np.zeros(self.dim)
        self.array[0] = 1  # |00...0⟩ state

    def entanglement_entropy(self, subsystem_size: int | None = None) -> float:
        """Calculate von Neumann entanglement entropy for a bipartition.
        
        Args:
            subsystem_size: Size of subsystem A (defaults to L//2)
            
        Returns:
            Von Neumann entropy in bits (base-2 logarithm)
        """
        if subsystem_size is None:
            subsystem_size = self.L // 2

        # Reshape state for proper bipartition
        dim_A = 2**subsystem_size
        dim_B = 2 ** (self.L - subsystem_size)
        state_flat = self.array.flatten()
        state_matrix = state_flat.reshape((dim_A, dim_B))

        _, s, _ = np.linalg.svd(state_matrix, full_matrices=False)

        schmidt_coeffs = s[s > 1e-12]  # Filter out numerical zeros

        entropy_terms = schmidt_coeffs**2 * np.log2(schmidt_coeffs**2)
        return -np.sum(entropy_terms)
