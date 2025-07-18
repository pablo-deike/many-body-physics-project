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
        # self.state.create_all_measurement_sigz()
        self.state.create_sigz_measurement()
        self.create_U_even_gate()
        self.create_U_odd_gate()

    def create_U_even_gate(self) -> None:
        I = np.eye(2)
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])
        self.U_even = np.kron(P0, scipy.stats.unitary_group(dim=2).rvs()) + np.kron(P1, I)
        ops = []
        for _ in range(self.L // 2):
            ops.append(self.U_even)
        global_op = ops[0]
        for op in ops[1:]:
            global_op = np.kron(global_op, op)
        self.U_global_even = np.reshape(global_op, [2, 2] * self.L)

    def create_U_odd_gate(self) -> None:
        I = np.eye(2)
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])
        self.U_odd = np.kron(I, P0) + np.kron(scipy.stats.unitary_group(dim=2).rvs(), P1)
        ops = []
        for _ in range(self.L // 2):
            ops.append(self.U_odd)
        global_op = ops[0]
        for op in ops[1:]:
            global_op = np.kron(global_op, op)
        self.U_global_odd = np.reshape(global_op, [2, 2] * self.L)

    def apply_U_gate(self, even: bool = True) -> None:
        """Apply the unitary operator U to the state."""
        if even:
            self.state.apply_U_gate(self.U_global_even)
        else:
            self.state.apply_U_gate(self.U_global_odd)

    def apply_gate(self, U: NDArray, r: int, unitary: bool = True) -> NDArray:
        l = len(U.shape) // 2
        intermediate_op = np.tensordot(self.state.array, U, axes=(range(r, r + l), range(l, 2 * l)))
        intermediate_op = np.transpose(
            intermediate_op, ([*range(r)] + [*range(self.L - l, self.L)] + [*range(r, self.L - l)])
        )
        if unitary is True:
            return intermediate_op
        return intermediate_op / np.linalg.norm(intermediate_op)

    def site_expectation_value(self, op: NDArray, site: int) -> float:
        """Calculate expectation value of a local operator."""
        expectation_right = self.apply_gate(op, site, unitary=True)
        expectation_left = np.vdot(self.state.array, expectation_right)
        return np.real_if_close(expectation_left)

    def apply_random_sigz_measurement(self) -> None:
        # prob = (1 + self.state.expectation_value(self.state.sigz_op[site])) / 2
        sigz = np.array([[1, 0], [0, -1]])
        mask = np.random.rand(self.L) < self.p
        sites = np.where(mask)[0]
        for site in sites:
            prob = (1 + self.site_expectation_value(sigz, site)) / 2
            if np.random.rand() < prob:
                self.state.array = self.apply_gate((1 + sigz) / 2, site, unitary=False)
            else:
                self.state.array = self.apply_gate((1 - sigz) / 2, site, unitary=False)

    def full_circuit_evolution(self, t: int) -> None:
        """Perform full circuit evolution for t time steps.
        This method applies the unitary operator U to pairs of sites and performs
        random measurements on the sites with a probability p.
        """
        I = np.eye(2)
        P0 = np.array([[1, 0], [0, 0]])
        P1 = np.array([[0, 0], [0, 1]])
        for _ in range(0, t, 2):
            U_odd = (np.kron(I, P0) + np.kron(scipy.stats.unitary_group(dim=2).rvs(), P1)).reshape([2, 2, 2, 2])
            U_even = (np.kron(P0, scipy.stats.unitary_group(dim=2).rvs()) + np.kron(P1, I)).reshape([2, 2, 2, 2])
            for site in range(0, self.L, 2):
                self.state.array = self.apply_gate(U_even, site)
            self.apply_random_sigz_measurement()
            for site in range(1, self.L-1, 2):
                self.state.array = self.apply_gate(U_odd, site)
            self.apply_random_sigz_measurement()