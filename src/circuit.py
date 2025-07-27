from copy import deepcopy
import numpy as np
import scipy.stats
from numpy.typing import NDArray
from src.state import State


class Circuit:
    """Quantum circuit for many-body physics simulations with random measurements."""
    
    def __init__(
        self,
        L: int,
        p: float,
        initial_state: State | None = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize the quantum circuit.
        
        Args:
            L: Number of sites in the system
            p: Probability of measurement at each site
            initial_state: Initial quantum state (defaults to all-zero state)
            random_seed: Seed for random number generation
        """
        self.L = L
        self.dim = 2**L
        self.p = p
        self.U = scipy.stats.unitary_group(dim=2, seed=random_seed).rvs()
        if initial_state is None:
            self.state = State(L)
        else:
            self.state = deepcopy(initial_state)

    def apply_gate(self, U: NDArray, r: int, unitary: bool = True) -> NDArray:
        """Apply a quantum gate to the state at position r.
        
        Args:
            U: Unitary operator to apply
            r: Starting site position for the gate
            unitary: Whether to normalize the result (False for measurements)
            
        Returns:
            Updated state array
        """
        l = len(U.shape) // 2
        if r == self.L - 1 and l == 2:
            # Special case for the last site
            intermediate_op = np.tensordot(self.state.array, U, axes=([self.L - 1, 0], range(l, 2 * l)))
            intermediate_op = np.transpose(intermediate_op, ([self.L - 1] + [*range(r)]))
        else:
            intermediate_op = np.tensordot(self.state.array, U, axes=(range(r, r + l), range(l, 2 * l)))
            intermediate_op = np.transpose(
                intermediate_op, ([*range(r)] + [*range(self.L - l, self.L)] + [*range(r, self.L - l)])
            )
        if unitary is True:
            return intermediate_op
        return intermediate_op / np.linalg.norm(intermediate_op)

    def site_expectation_value(self, op: NDArray, site: int) -> float:
        """Calculate expectation value of a local operator.
        
        Args:
            op: Local operator to compute expectation value for
            site: Site index where to apply the operator
            
        Returns:
            Real expectation value
        """
        expectation_right = self.apply_gate(op, site, unitary=True)
        expectation_left = np.vdot(self.state.array, expectation_right)
        return np.real_if_close(expectation_left)

    def apply_random_sigz_measurement(self) -> None:
        """Apply random projective measurements in the Z-basis.
        
        Each site is measured with probability p. The measurement outcome
        is determined probabilistically based on the expectation value of σz.
        """
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
        
        Each time step consists of:
        1. Apply random 2-qubit gates to even pairs of sites
        2. Apply random measurements with probability p
        3. Apply random 2-qubit gates to odd pairs of sites  
        4. Apply random measurements with probability p
        
        Args:
            t: Number of time steps to evolve
        """
        for _ in range(t):
            U_random = scipy.stats.unitary_group(dim=4).rvs()
            U_gate = U_random.reshape([2, 2, 2, 2])

            # Apply gates to even pairs
            for site in range(0, self.L, 2):
                self.state.array = self.apply_gate(U_gate, site)

            self.apply_random_sigz_measurement()

            # Apply gates to odd pairs
            for site in range(1, self.L + 1, 2):
                self.state.array = self.apply_gate(U_gate, site % self.L)

            self.apply_random_sigz_measurement()
