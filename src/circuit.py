from copy import deepcopy
import numpy as np
import scipy.stats
from numpy.typing import NDArray
from src.state import State


class Circuit:
    def __init__(
        self,
        L: int,
        p: float,
        initial_state: State | None = None,
        random_seed: int = 42,
    ) -> None:
        self.L = L
        self.dim = 2**L
        self.p = p
        self.U = scipy.stats.unitary_group(dim=2, seed=random_seed).rvs()
        if initial_state is None:
            self.state = State(L)
        else:
            self.state = deepcopy(initial_state)
        self.state.create_all_measurement_sigz()
        self.create_U_even_gate()
        self.create_U_odd_gate()

    def create_U_even_gate(self) -> None:
        I = np.eye(2)
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])
        CU = np.kron(P0, scipy.stats.unitary_group(dim=2).rvs()) + np.kron(P1, I)
        ops = []
        for _ in range(self.L // 2):
            ops.append(CU)
        global_op = ops[0]
        for op in ops[1:]:
            global_op = np.kron(global_op, op)
        self.U_even = global_op

    def create_U_odd_gate(self) -> None:
        I = np.eye(2)
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])
        CU = np.kron(I, P0) + np.kron(scipy.stats.unitary_group(dim=2).rvs(), P1)
        ops = []
        for _ in range(self.L // 2):
            ops.append(CU)
        global_op = ops[0]
        for op in ops[1:]:
            global_op = np.kron(global_op, op)
        self.U_odd = global_op

    def apply_U_gate(self, even: bool = True) -> None:
        """Apply the unitary operator U to the state."""
        if even:
            self.state.apply_U_gate(self.U_even)
        else:
            self.state.apply_U_gate(self.U_odd)

    def apply_gate(self, U: NDArray, r: int, unitary: bool = True) -> None:
        l = len(U.shape) // 2
        self.state.array = np.tensordot(self.state.array, U, axes=(range(r, r + l), range(l, 2 * l)))
        self.state.array = np.transpose(
            self.state.array, ([*range(r)] + [*range(self.L - l, self.L)] + [*range(r, self.L - l)])
        )
        if unitary is True:
            return
        self.state.array = self.state.array / np.linalg.norm(self.state.array)

    def apply_random_sigz_measurement(self, site: int) -> None:
        prob = self.state.expectation_value(self.state.sigz_op[site])
        if np.random.rand() < self.p:
            if np.random.rand() < prob:
                self.state.apply_site_operator((1 + self.state.sigz_op[site].toarray()) / 2)
            else:
                self.state.apply_site_operator((1 - self.state.sigz_op[site].toarray()) / 2)

    def full_circuit_evolution(self, t: int) -> None:
        """Perform full circuit evolution for t time steps.
        This method applies the unitary operator U to pairs of sites and performs
        random measurements on the sites with a probability p.
        """
        for _ in range(0, t, 2):
            self.apply_U_gate(even=True)
            for site in range(self.L):
                self.apply_random_sigz_measurement(site)
            self.apply_U_gate(even=False)
            for site in range(self.L):
                self.apply_random_sigz_measurement(site)
