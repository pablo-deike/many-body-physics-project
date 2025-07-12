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

    def init_all_zeros(self) -> None:
        # maybe it´s a good idea for the future to considere it in a binary state
        self.array = np.zeros(self.dim)
        self.array[0] = 1  # |00...0⟩ state

    def create_all_measurement_sigz(self) -> None:
        sigz = np.array([[1, 0], [0, -1]])
        all_ops = []
        for site in range(self.L):
            ops = []
            for i in range(self.L):
                if i == site:
                    ops.append(csr_matrix(sigz))
                else:
                    ops.append(identity(2, format="csr"))
            sigz_op = ops[0]
            for op in ops[1:]:
                sigz_op = kron(sigz_op, op, format="csr")
            all_ops.append(sigz_op)
        self.sigz_op = all_ops

    def expectation_value(self, op: NDArray) -> float:
        """Calculate expectation value of a local operator."""
        return np.real(np.vdot(self.array, op @ self.array))

    def apply_site_operator(self, op: NDArray) -> None:
        if not hasattr(self, "sigz_op"):
            raise ValueError("sigz operator not defined. Call measure_sigz first.")
        self.array = op @ self.array
        self.array = self.array.flatten()
        self.array = self.array / np.linalg.norm(self.array)

    def apply_U_gate(self, U: np.ndarray) -> None:
        self.array = U @ self.array

    def entanglement_entropy(self, subsystem_size: int | None = None) -> float:  # hacerlo con SVD
        """Calculate von Neumann entanglement entropy for a bipartition"""
        if subsystem_size is None:
            subsystem_size = self.L // 2
        dim_A = 2**subsystem_size
        dim_B = 2 ** (self.L - subsystem_size)
        total_density_matrix = self.construct_density_matrix().reshape((dim_A, dim_B, dim_A, dim_B))

        # Compute reduced density matrix by tracing out subsystem B
        # ρ_A = state_matrix @ state_matrix.conj().T
        rho_A = np.trace(total_density_matrix, axis1=1, axis2=3)

        # Get eigenvalues of reduced density matrix
        eigenvalues = np.real_if_close(np.linalg.eigvalsh(rho_A))
        eigenvalues = eigenvalues[eigenvalues > 1e-12]

        # Calculate von Neumann entropy
        diag = np.diag(rho_A)
        # Avoid log2(0) by setting log2(0) = 0 for zero diagonal elements
        entropy_terms = np.where(diag > 0, diag * np.log2(diag), 0)
        return np.real_if_close(-np.sum(entropy_terms))

    def construct_density_matrix(self) -> NDArray:
        return np.outer(self.array, np.conj(self.array))
