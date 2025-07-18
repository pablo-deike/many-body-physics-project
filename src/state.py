import numpy as np
from scipy.sparse import kron, identity, csr_matrix
from numpy.typing import ArrayLike, NDArray


class State:
    def __init__(self, L: int, array: ArrayLike | None = None) -> None:
        self.dim = 2**L
        self.L = L
        if array is None:
            self.init_all_zeros()
        else:
            if len(array) != self.dim:
                raise ValueError(f"State must have length {self.dim}, got {len(array)}")
            self.array = np.array(array, dtype=np.complex128)
        self.array = self.array.flatten()
        self.array = np.reshape(self.array, [2]*L)

    def init_all_zeros(self) -> None:
        # maybe it´s a good idea for the future to considere it in a binary state
        self.array = np.zeros(self.dim)
        self.array[0] = 1  # |00...0⟩ state

    # def create_all_measurement_sigz(self) -> None:
    #     sigz = np.array([[1, 0], [0, -1]])
    #     all_ops = []
    #     for site in range(self.L):
    #         ops = []
    #         for i in range(self.L):
    #             if i == site:
    #                 ops.append(csr_matrix(sigz))
    #             else:
    #                 ops.append(identity(2, format="csr"))
    #         sigz_op = ops[0]
    #         for op in ops[1:]:
    #             sigz_op = kron(sigz_op, op, format="csr")
    #         all_ops.append(sigz_op)
    #     self.sigz_op = all_ops

    def create_sigz_measurement(self) -> None:
        sigz = np.array([[1, 0], [0, -1]])
        global_op = sigz
        for i in range(self.L- 1):
            global_op = np.kron(global_op, sigz)
        self.sigz_op = np.reshape(global_op, [2, 2]*self.L)

    def expectation_value(self, op: NDArray) -> float:
        """Calculate expectation value of a local operator."""
        return np.real(np.vdot(self.array, op @ self.array))

    def apply_site_operator(self, op: NDArray, site: int) -> None:
        if not hasattr(self, "sigz_op"):
            raise ValueError("sigz operator not defined. Call measure_sigz first.")
        op_site = np.tensordot(self.array, op, axes=([site], [1])) # ... [j] ... i [i*]
        op_site = np.transpose(op_site, ([*range(site)] + [*range(self.L - 1, self.L)] + [*range(site, self.L - 1)]))
        self.array = op_site / np.linalg.norm(op_site)

    def apply_U_gate(self, U: NDArray) -> None:
        self.array = U @ self.array

    def entanglement_entropy(self, subsystem_size: int | None = None) -> float:
        """Calculate von Neumann entanglement entropy for a bipartition"""
        if subsystem_size is None:
            subsystem_size = self.L // 2
        
        # Reshape state for proper bipartition
        dim_A = 2**subsystem_size
        dim_B = 2**(self.L - subsystem_size)
        
        # Flatten the state array and reshape for bipartition
        state_flat = self.array.flatten()
        state_matrix = state_flat.reshape((dim_A, dim_B))
        
        # Compute reduced density matrix by SVD (more numerically stable)
        U, s, Vh = np.linalg.svd(state_matrix, full_matrices=False)
        
        # Schmidt coefficients are the singular values
        schmidt_coeffs = s[s > 1e-12]  # Filter out numerical zeros
        
        # Calculate von Neumann entropy from Schmidt coefficients
        entropy_terms = schmidt_coeffs**2 * np.log2(schmidt_coeffs**2)
        return -np.sum(entropy_terms)

    def construct_density_matrix(self) -> NDArray:
        return np.outer(np.conj(self.array), self.array)
