from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, List
import warnings

# ── Attempt to import Nelson's CD library ──────────────────────────────────────
try:
    from cd import (
        solve_1d_picard,
        viability_threshold_1d,
        principal_eigenvalue_1d,
        check_convergence,
        solution_type,
    )
    CD_AVAILABLE = True
except ImportError:
    CD_AVAILABLE = False
    warnings.warn(
        "navi-creative-determinant not found. "
        "Install from Project-Navi/navi-creative-determinant. "
        "CDStateUpdater will fall back to GRU.",
        ImportWarning,
        stacklevel=2,
    )


# ── Global dimensions — must match Fabric notebook ────────────────────────────
D_STATE = 64   # Module hidden state dimension
D_MSG   = 64   # Encoded message dimension
D_QUERY = 32   # Goal query dimension

# CD PDE domain parameters
# D_STATE=64 → N=64 grid points over L=1.0 domain
# Each state dimension maps to one spatial grid point
CD_N    = D_STATE   # 1D grid resolution = state dimension
CD_L    = 1.0       # Domain [0, L]
CD_C    = 10.0      # Nonlinearity exponent (default from Nelson's quick start)
CD_P    = 2.0       # PDE p-parameter


@dataclass
class CDUpdateRecord:
    """Records the outcome of a single CD PDE state update."""
    step:            int
    module_id:       str
    converged:       bool
    solution_type:   Literal['trivial', 'nontrivial']
    max_phi:         float
    beta_used:       float
    beta_star:       float
    above_threshold: bool
    iterations:      int
    residual:        float

    @property
    def bifurcated(self) -> bool:
        """True if the module transitioned to a nontrivial state this step."""
        return self.solution_type == 'nontrivial'


class CDStateUpdater:
    """
    CD PDE state update — drop-in replacement for GRUCell.

    Maps the compressed Goal-Gate message to viability potential b,
    the goal query to creative drive a, and solves the 1D Picard
    iteration to find the new module state Φ.

    The solution Φ ∈ R^N (N = D_STATE = 64) is the new state vector.

    Mathematical guarantee:
        MonotoneFixedPoint.lean — convergence to unique fixed point.
        Below viability threshold β*: trivial solution (state unchanged).
        Above viability threshold β*: nontrivial solution (state transitions).

    Parameters
    ----------
    d_state : int
        Module state dimension. Must match D_STATE. Default 64.
    L : float
        PDE domain length. Default 1.0.
    c : float
        Nonlinearity exponent. Default 10.0.
    beta_scale : float
        Scales the compressed message to viability potential.
        Controls how strongly incoming messages drive state transitions.
        Start at 1.2 * beta_star (above threshold) for active modules.
    fallback_to_gru : bool
        If True and CD library unavailable, use GRU silently.
        If False, raise on CD unavailability.
    """

    def __init__(
        self,
        d_state:         int   = D_STATE,
        L:               float = CD_L,
        c:               float = CD_C,
        beta_scale:      float = 1.2,
        fallback_to_gru: bool  = True,
    ):
        self.d_state         = d_state
        self.N               = d_state   # grid points = state dim
        self.L               = L
        self.c               = c
        self.beta_scale      = beta_scale
        self.fallback_to_gru = fallback_to_gru
        self.records: List[CDUpdateRecord] = []

        if not CD_AVAILABLE and not fallback_to_gru:
            raise ImportError(
                "navi-creative-determinant required but not installed. "
                "Set fallback_to_gru=True or install the library."
            )

        # Precompute viability threshold for constant b=1.0
        # Actual threshold scales with the message norm
        if CD_AVAILABLE:
            self._beta_star_unit = viability_threshold_1d(self.L, b=1.0)
        else:
            self._beta_star_unit = None

    def _message_to_viability(self, compressed: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Map compressed Goal-Gate message to viability field b(x).

        The message vector (D_MSG=64) maps directly to the 1D grid (N=64).
        b(x) = |compressed| normalized to [0, 1] — represents what the
        receiving module can sustain given the incoming signal.

        Returns
        -------
        b : np.ndarray shape (N,)
            Viability potential field.
        b_mean : float
            Mean viability (scalar summary for threshold computation).
        """
        msg = compressed.flatten()[:self.N]
        # Normalize to [0, 1] — viability is a positive bounded field
        msg_min, msg_max = msg.min(), msg.max()
        if msg_max - msg_min < 1e-8:
            b = np.ones(self.N) * 0.5
        else:
            b = (msg - msg_min) / (msg_max - msg_min)
        return b, float(b.mean())

    def _query_to_drive(self, goal_query: np.ndarray) -> np.ndarray:
        """
        Map goal query q_j to creative drive a(x).

        The goal query (D_QUERY=32) is upsampled to the 1D grid (N=64).
        a(x) represents what the module is trying to become — the creative
        tension driving it toward a nontrivial state.

        Returns
        -------
        a : np.ndarray shape (N,)
            Creative drive field.
        """
        q = goal_query.flatten()[:D_QUERY]
        # Upsample D_QUERY=32 → N=64 via linear interpolation
        x_q = np.linspace(0, 1, len(q))
        x_n = np.linspace(0, 1, self.N)
        a = np.interp(x_n, x_q, q)
        # Creative drive is non-negative
        a = np.abs(a)
        return a

    def update(
        self,
        compressed:  torch.Tensor,
        state:       torch.Tensor,
        goal_query:  torch.Tensor,
        step:        int = 0,
        module_id:   str = '?',
    ) -> torch.Tensor:
        """
        Compute new module state via CD PDE Picard iteration.

        Parameters
        ----------
        compressed : torch.Tensor (1, D_MSG)
            Goal-Gate compressed message M(i→j).
        state : torch.Tensor (1, D_STATE)
            Current module state.
        goal_query : torch.Tensor (1, D_QUERY)
            Receiver's goal query q_j.
        step : int
            Current propagation step (for record keeping).
        module_id : str
            Module identifier (for record keeping).

        Returns
        -------
        new_state : torch.Tensor (1, D_STATE)
            Updated module state = Φ solution of CD PDE.
        """
        if not CD_AVAILABLE:
            # Should not reach here if fallback_to_gru=True and GRU is used upstream
            raise RuntimeError("CD library unavailable and fallback disabled.")

        # Convert to numpy
        msg_np   = compressed.detach().numpy().flatten()
        query_np = goal_query.detach().numpy().flatten()

        # Map to PDE fields
        b_field, b_mean = self._message_to_viability(msg_np)
        a_field         = self._query_to_drive(query_np)

        # Compute viability threshold for this b field
        beta_star = float(self._beta_star_unit) * b_mean if b_mean > 1e-8 else float(self._beta_star_unit)

        # Set beta: scale above threshold to drive nontrivial solutions
        # beta_scale=1.2 means 20% above threshold → nontrivial
        beta_b = self.beta_scale * beta_star * b_mean

        # Solve CD PDE via Picard iteration
        try:
            x, Phi, info = solve_1d_picard(
                L      = self.L,
                N      = self.N,
                a      = float(a_field.mean()),   # scalar a for 1D
                beta_b = beta_b,
                c      = self.c,
            )

            converged   = check_convergence(info)[0]
            sol_type    = solution_type(info)
            max_phi     = float(info.get('maxPhi', np.abs(Phi).max()))
            n_iter      = int(info.get('iterations', -1))
            residual    = float(info.get('residual', -1.0))

            # Phi is the new state: shape (N,) = (D_STATE,)
            # Normalize to same scale as incoming state
            state_norm = float(torch.norm(state).item())
            phi_norm   = float(np.linalg.norm(Phi))
            if phi_norm > 1e-8 and state_norm > 1e-8:
                Phi_scaled = Phi * (state_norm / phi_norm)
            else:
                Phi_scaled = Phi

            new_state = torch.tensor(
                Phi_scaled, dtype=torch.float32
            ).unsqueeze(0)  # (1, D_STATE)

        except Exception as e:
            warnings.warn(
                f"CDStateUpdater: Picard solver failed at step {step} "
                f"module {module_id}: {e}. Returning current state.",
                RuntimeWarning,
                stacklevel=2,
            )
            new_state   = state
            converged   = False
            sol_type    = 'trivial'
            max_phi     = 0.0
            beta_star   = 0.0
            n_iter      = 0
            residual    = float('inf')

        # Record the update
        self.records.append(CDUpdateRecord(
            step            = step,
            module_id       = module_id,
            converged       = converged,
            solution_type   = sol_type,
            max_phi         = max_phi,
            beta_used       = beta_b,
            beta_star       = beta_star,
            above_threshold = beta_b > beta_star,
            iterations      = n_iter,
            residual        = residual,
        ))

        return new_state

    def summary(self) -> dict:
        """Summary statistics across all recorded updates."""
        if not self.records:
            return {}
        n_total        = len(self.records)
        n_converged    = sum(1 for r in self.records if r.converged)
        n_nontrivial   = sum(1 for r in self.records if r.bifurcated)
        n_above        = sum(1 for r in self.records if r.above_threshold)
        return {
            'total_updates':      n_total,
            'converged':          n_converged,
            'convergence_rate':   n_converged / n_total,
            'nontrivial':         n_nontrivial,
            'bifurcation_rate':   n_nontrivial / n_total,
            'above_threshold':    n_above,
            'mean_max_phi':       np.mean([r.max_phi for r in self.records]),
            'mean_residual':      np.mean([r.residual for r in self.records
                                           if r.residual != float('inf')]),
        }


class FabricStateUpdater(nn.Module):
    """
    A/B state updater for FabricModule.

    Wraps both GRU and CDStateUpdater behind a single interface.
    Set use_cd=True to activate the CD PDE update.
    Set use_cd=False for GRU baseline.

    This A/B design is the core Phase 2 experimental contribution:
    same Fabric, same SAD probe, same goal-gating — two state updaters.
    Compare attractor geometry, convergence rate, bifurcation behavior.

    Parameters
    ----------
    d_msg : int
        Message dimension. Default D_MSG=64.
    d_state : int
        State dimension. Default D_STATE=64.
    d_query : int
        Query dimension. Default D_QUERY=32.
    use_cd : bool
        If True, use CD PDE. If False, use GRU. Default False (safe start).
    beta_scale : float
        CD viability scale. Default 1.2 (20% above threshold).
    """

    def __init__(
        self,
        d_msg:      int   = D_MSG,
        d_state:    int   = D_STATE,
        d_query:    int   = D_QUERY,
        use_cd:     bool  = False,
        beta_scale: float = 1.2,
    ):
        super().__init__()
        self.use_cd   = use_cd
        self.d_state  = d_state

        # GRU — always available, baseline
        self.gru = nn.GRUCell(d_msg, d_state)

        # CD updater — requires Nelson's library
        if CD_AVAILABLE:
            self.cd = CDStateUpdater(
                d_state         = d_state,
                beta_scale      = beta_scale,
                fallback_to_gru = True,
            )
        else:
            self.cd = None
            if use_cd:
                warnings.warn(
                    "use_cd=True but navi-creative-determinant not installed. "
                    "Falling back to GRU. Install cd library to enable CD updates.",
                    UserWarning,
                    stacklevel=2,
                )
                self.use_cd = False

    def forward(
        self,
        compressed:  torch.Tensor,
        state:       torch.Tensor,
        goal_query:  torch.Tensor,
        step:        int = 0,
        module_id:   str = '?',
    ) -> torch.Tensor:
        """
        Update module state.

        Parameters
        ----------
        compressed : torch.Tensor (1, D_MSG)
        state      : torch.Tensor (1, D_STATE)
        goal_query : torch.Tensor (1, D_QUERY)
        step       : int
        module_id  : str

        Returns
        -------
        new_state : torch.Tensor (1, D_STATE)
        """
        if self.use_cd and self.cd is not None:
            return self.cd.update(compressed, state, goal_query, step, module_id)
        else:
            # GRU baseline — Lemma 4.2 (empirical)
            return self.gru(compressed, state)

    def cd_summary(self) -> dict:
        """CD update statistics. Empty if running GRU mode."""
        if self.cd is not None:
            return self.cd.summary()
        return {}

    def switch_to_cd(self):
        """Activate CD PDE update mode."""
        if not CD_AVAILABLE:
            raise RuntimeError("navi-creative-determinant not installed.")
        self.use_cd = True

    def switch_to_gru(self):
        """Activate GRU baseline mode."""
        self.use_cd = False


# ── Integration patch for FabricModule ────────────────────────────────────────
# Replace in FabricModule.__init__():
#
#   OLD:
#       self.gru = nn.GRUCell(d_msg, d_state)
#
#   NEW:
#       self.state_updater = FabricStateUpdater(
#           d_msg=d_msg, d_state=d_state, d_query=d_query,
#           use_cd=False,  # start GRU, switch to CD for Phase 2
#       )
#
# Replace in FabricModule.receive_message():
#
#   OLD:
#       new_state = self.gru(compressed, state)
#
#   NEW:
#       new_state = self.state_updater(
#           compressed, state, q_j,
#           step=step, module_id=self.module_id
#       )
#
# Phase 2 A/B experiment:
#
#   # GRU baseline run
#   fabric_gru = OGIFabric(use_cd=False)
#   results_gru = fabric_gru.propagate(n_steps=40)
#
#   # CD PDE run
#   fabric_cd = OGIFabric(use_cd=True)
#   results_cd = fabric_cd.propagate(n_steps=40)
#
#   # Compare attractor geometry — same SAD probe, same goal-gating
#   # GRU: empirical convergence (Lemma 4.2)
#   # CD:  verified convergence (MonotoneFixedPoint.lean)


if __name__ == '__main__':
    print("FabricStateUpdater — integration test")
    print(f"  CD library available: {CD_AVAILABLE}")
    print(f"  D_STATE={D_STATE}, D_MSG={D_MSG}, D_QUERY={D_QUERY}")
    print(f"  CD grid: N={CD_N}, L={CD_L}, c={CD_C}")
    print()

    # Test with synthetic tensors
    compressed  = torch.randn(1, D_MSG)
    state       = torch.randn(1, D_STATE)
    goal_query  = torch.randn(1, D_QUERY)

    # GRU mode
    updater_gru = FabricStateUpdater(use_cd=False)
    new_state_gru = updater_gru(compressed, state, goal_query, step=0, module_id='A')
    print(f"  GRU output shape:  {new_state_gru.shape}")
    print(f"  GRU output norm:   {torch.norm(new_state_gru).item():.4f}")

    # CD mode (if available)
    if CD_AVAILABLE:
        updater_cd = FabricStateUpdater(use_cd=True)
        new_state_cd = updater_cd(compressed, state, goal_query, step=0, module_id='A')
        print(f"  CD output shape:   {new_state_cd.shape}")
        print(f"  CD output norm:    {torch.norm(new_state_cd).item():.4f}")
        print(f"  CD summary:        {updater_cd.cd_summary()}")
    else:
        print("  CD mode: library not installed — install navi-creative-determinant")

    print()
    print("Integration patch:")
    print("  FabricModule.gru → FabricModule.state_updater")
    print("  new_state = self.state_updater(compressed, state, q_j, step, module_id)")
