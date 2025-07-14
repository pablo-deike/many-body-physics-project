from copy import deepcopy
import numpy as np
import scipy.stats
from numpy.typing import NDArray
from src.mps import MPS, init_spinup_MPS, split_truncate_theta, svd


class Circuit:
    def __init__(
        self,
        L: int,
        p: float,
        initial_state: MPS | None = None,
        random_seed: int = 42,
    ) -> None:
        self.L = L
        self.dim = 2**L
        self.p = p
        self.U = scipy.stats.unitary_group(dim=2, seed=random_seed).rvs()
        if initial_state is None:
            self.state = init_spinup_MPS(L)
        else:
            self.state = deepcopy(initial_state)
        self.create_U_even_gate()
        self.create_U_odd_gate()

    def create_U_even_gate(self) -> None:
        I = np.eye(2)
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])
        CU = np.kron(P0, scipy.stats.unitary_group(dim=2).rvs()) + np.kron(P1, I)
        self.U_even = np.reshape(CU, [2, 2, 2, 2])  # i i* j j*

    def create_U_odd_gate(self) -> None:
        I = np.eye(2)
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])
        CU = np.kron(I, P0) + np.kron(scipy.stats.unitary_group(dim=2).rvs(), P1)
        self.U_odd = np.reshape(CU, [2, 2, 2, 2])  # i i* j j*

    def apply_U_gate(self, even: bool = True) -> None:
        """Apply the unitary operator U to the state."""
        # self.apply_gate(scipy.stats.unitary_group(dim=2).rvs(), r=0)
        U = (
            self.U_even 
        )  # this might be unneccessary with the tensor dots one only needs one gate and collapsing the indices
        chi_max = 32
        for site in range(0, self.L, 2 if even else 1):
            j = (site + 1) % self.L
            op_theta = np.tensordot(
                U, self.state.get_theta2(site), axes=([2, 3], [1, 2])
            )  # i j [i*] [j*] vL [i] [j] vR* -> i j vL vR*
            op_theta = np.transpose(op_theta, (2, 0, 1, 3))
            Ai, Sj, Bj = split_truncate_theta(op_theta, chi_max, 1e-10)
            # put back into MPS
            Gi = np.tensordot(np.diag(self.state.Ss[site] ** (-1)), Ai, axes=[1, 0])  # vL [vL*], [vL] i vC
            self.state.Bs[site] = np.tensordot(Gi, np.diag(Sj), axes=[2, 0])  # vL i [vC], [vC] vC
            self.state.Ss[j] = Sj  # vC
            self.state.Bs[j] = Bj  # vC j vR

    def apply_gate(self, U: NDArray, r: int, unitary: bool = True) -> None:
        """Apply a two qubit operator U to the state at sites r to r+l.
        The operator U has shape [2, 2, ..., 2]"""
        l = len(U.shape) // 2
        op_theta = np.tensordot(self.state.get_theta2(r), U, axes=(range(r, r + l), range(l, 2 * l)))
        op_theta = np.transpose(op_theta, ([*range(r)] + [*range(self.L - l, self.L)] + [*range(r, self.L - l)]))
        self.state.Bs[r], self.state.Ss[r + 1], self.state.Bs[r + 1] = split_truncate_theta(op_theta, self.dim, 1e-10)
        if unitary is True:
            return
        self.state.Bs[r] = self.state.Bs[r] / np.linalg.norm(self.state.Bs[r])
        self.state.Bs[r + 1] = self.state.Bs[r + 1] / np.linalg.norm(self.state.Bs[r + 1])

    def apply_random_sigz_measurement(self) -> None:
        sigz = np.array([[1, 0], [0, -1]])
        probs = (1 + self.state.site_expectation_value(sigz)) / 2
        mask = np.random.rand(self.L) < self.p
        sites = np.where(mask)[0]
        for site in sites:
            if np.random.rand() < probs[site]:
                self.state.apply_operator((1 + sigz) / 2, site)
                # self.state.Bs[site] = ((1 + sigz) / 2) @ self.state.Bs[site]
            else:
                self.state.apply_operator((1 - sigz) / 2, site)
                # self.state.Bs[site] = ((1 - sigz) / 2) @ self.state.Bs[site]

    def full_circuit_evolution(self, t: int) -> None:
        """Perform full circuit evolution for t time steps.
        This method applies the unitary operator U to pairs of sites and performs
        random measurements on the sites with a probability p.
        """
        for _ in range(0, t, 2):
            self.apply_U_gate(even=True)
            self.apply_random_sigz_measurement()

            self.apply_U_gate(even=False)
            self.apply_random_sigz_measurement()
