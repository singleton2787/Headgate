"""
OGI Fabric — CD PDE State Updater v2
======================================
Drop-in replacement for the GRU state update in FabricModule.receive_message().

v2 fix: spatially varying beta_b array instead of scalar.
Scalar beta_b causes all modules to converge to the same PDE solution
regardless of goal vector — complete collapse of inter-module diversity.
Array beta_b modulates viability per spatial location using the goal query,
producing genuinely different attractor states per module.

Mathematical grounding:
  GRU:    empirical convergence, Lemma 4.2
  CD PDE: MonotoneFixedPoint.lean — 15 theorems, zero sorry
          Convergence to unique fixed point VERIFIED.
          Array beta_b gives goal-differentiated attractors.

Corten Research — Singleton, Dollinger & Spence
arXiv:2411.15832 Phase 2
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, List
import warnings

try:
    from cd import solve_1d_picard, viability_threshold_1d, check_convergence, solution_type
    CD_AVAILABLE = True
except ImportError:
    CD_AVAILABLE = False
    warnings.warn("navi-creative-determinant not found. Using GRU only.", ImportWarning, stacklevel=2)

D_STATE = 64
D_MSG   = 64
D_QUERY = 32


@dataclass
class CDUpdateRecord:
    step: int; module_id: str; converged: bool
    solution_type: Literal['trivial', 'nontrivial']
    max_phi: float; beta_mean: float; beta_star: float
    above_threshold: bool; iterations: int
    @property
    def bifurcated(self): return self.solution_type == 'nontrivial'


class CDStateUpdater:
    """
    CD PDE state update with spatially varying beta_b.

    Key fix from v1: beta_b is now an array shaped (N,) where each
    spatial location is modulated by both the message viability field
    AND the receiver's goal vector. This produces goal-differentiated
    attractor states across modules — the prerequisite for Gater integration.

    beta_b(x) = beta_scale * beta_star * b_field(x) * goal_mod(x)

    b_field:   message → viability potential [0, 1]
    goal_mod:  goal query → spatial modulation [0.5, 1.5]
    """

    def __init__(self, d_state=D_STATE, L=1.0, c=10.0, beta_scale=1.2):
        self.d_state    = d_state
        self.N          = d_state
        self.L          = L
        self.c          = c
        self.beta_scale = beta_scale
        self.records: List[CDUpdateRecord] = []
        if CD_AVAILABLE:
            self._beta_star_unit = viability_threshold_1d(L, b=1.0)

    def _message_to_viability(self, msg_np):
        m = msg_np.flatten()[:self.N]
        lo, hi = m.min(), m.max()
        b = np.ones(self.N) * 0.5 if hi - lo < 1e-8 else (m - lo) / (hi - lo)
        return b, float(b.mean())

    def _query_to_goal_mod(self, query_np):
        """
        Map goal query to spatial modulation field [0.5, 1.5].
        Values above 1.0 amplify viability, below 1.0 suppress it.
        This is what differentiates modules with different goals.
        """
        q = query_np.flatten()[:D_QUERY]
        # Upsample D_QUERY=32 → N=64
        g = np.interp(np.linspace(0, 1, self.N), np.linspace(0, 1, len(q)), q)
        g_min, g_max = g.min(), g.max()
        if g_max - g_min < 1e-8:
            return np.ones(self.N)
        # Normalize to [0.5, 1.5] — amplifies or suppresses per location
        return 0.5 + (g - g_min) / (g_max - g_min)

    def _query_to_drive(self, query_np):
        q = query_np.flatten()[:D_QUERY]
        a = np.interp(np.linspace(0, 1, self.N), np.linspace(0, 1, len(q)), q)
        return np.abs(a)

    def update(self, compressed, state, goal_query, step=0, module_id='?'):
        msg_np   = compressed.detach().numpy().flatten()
        query_np = goal_query.detach().numpy().flatten()

        b_field, b_mean = self._message_to_viability(msg_np)
        a_field         = self._query_to_drive(query_np)
        goal_mod        = self._query_to_goal_mod(query_np)

        # Viability threshold for this message
        beta_star = viability_threshold_1d(self.L, b=b_mean) if b_mean > 1e-8 \
                    else float(self._beta_star_unit)

        # Spatially varying beta_b — goal modulates viability per location
        # This is the critical fix: each module gets a unique viability landscape
        beta_b_array = self.beta_scale * beta_star * b_field * goal_mod
        beta_b_mean  = float(beta_b_array.mean())

        try:
            x, Phi, info = solve_1d_picard(
                L      = self.L,
                N      = self.N,
                a      = float(a_field.mean()),
                beta_b = beta_b_array,   # ARRAY — goal-differentiated
                c      = self.c,
            )
            converged = check_convergence(info)[0]
            sol_type  = solution_type(info)
            max_phi   = float(info.get('maxPhi', np.abs(Phi).max()))
            n_iter    = int(info.get('iters', info.get('iterations', -1)))

            # Hard clamp to N interior points
            Phi = Phi.flatten()[:self.N + 2]
            if len(Phi) == self.N + 2:
                Phi = Phi[1:-1]
            Phi = Phi[:self.N]

            # Normalize to incoming state scale
            state_norm = float(torch.norm(state).item())
            phi_norm   = float(np.linalg.norm(Phi))
            if phi_norm > 1e-8 and state_norm > 1e-8:
                Phi = Phi * (state_norm / phi_norm)

            new_state = torch.tensor(Phi, dtype=torch.float32).unsqueeze(0)

        except Exception as e:
            warnings.warn(f"CD solver failed step={step} module={module_id}: {e}")
            new_state = state
            converged = False; sol_type = 'trivial'; max_phi = 0.0
            beta_star = 0.0; n_iter = 0; beta_b_mean = 0.0

        self.records.append(CDUpdateRecord(
            step=step, module_id=module_id, converged=converged,
            solution_type=sol_type, max_phi=max_phi,
            beta_mean=beta_b_mean, beta_star=beta_star,
            above_threshold=beta_b_mean > beta_star, iterations=n_iter,
        ))
        return new_state

    def summary(self):
        if not self.records: return {}
        n = len(self.records)
        return {
            'total_updates':    n,
            'converged':        sum(1 for r in self.records if r.converged),
            'convergence_rate': sum(1 for r in self.records if r.converged) / n,
            'nontrivial':       sum(1 for r in self.records if r.bifurcated),
            'bifurcation_rate': sum(1 for r in self.records if r.bifurcated) / n,
            'mean_max_phi':     np.mean([r.max_phi for r in self.records]),
        }


class FabricStateUpdater(nn.Module):
    """
    A/B state updater — GRU baseline vs CD PDE v2.

    use_cd=False  →  GRU (Lemma 4.2, empirical)
    use_cd=True   →  CD PDE v2 (spatially varying beta_b, goal-differentiated)

    Two line changes to FabricModule:

    __init__:
        self.state_updater = FabricStateUpdater(
            d_msg=d_msg, d_state=d_state, d_query=d_query, use_cd=False
        )

    receive_message:
        new_state = self.state_updater(
            compressed, state, q_j, step=0, module_id=self.module_id
        )
        new_state = new_state[:, :state.shape[-1]]  # shape clamp
    """

    def __init__(self, d_msg=D_MSG, d_state=D_STATE, d_query=D_QUERY,
                 use_cd=False, beta_scale=1.2):
        super().__init__()
        self.use_cd  = use_cd
        self.d_state = d_state
        self.gru = nn.GRUCell(d_msg, d_state)
        self.cd  = CDStateUpdater(d_state=d_state, beta_scale=beta_scale) \
                   if CD_AVAILABLE else None
        if use_cd and not CD_AVAILABLE:
            warnings.warn("use_cd=True but CD not installed. Falling back to GRU.")
            self.use_cd = False

    def forward(self, compressed, state, goal_query, step=0, module_id='?'):
        if self.use_cd and self.cd is not None:
            return self.cd.update(compressed, state, goal_query, step, module_id)
        return self.gru(compressed, state)

    def cd_summary(self): return self.cd.summary() if self.cd else {}
    def switch_to_cd(self):
        if not CD_AVAILABLE: raise RuntimeError("CD not installed.")
        self.use_cd = True
    def switch_to_gru(self): self.use_cd = False


if __name__ == '__main__':
    print(f"CD available: {CD_AVAILABLE}")
    compressed = torch.randn(1, D_MSG)
    state      = torch.randn(1, D_STATE)
    query      = torch.randn(1, D_QUERY)

    u_gru = FabricStateUpdater(use_cd=False)
    s_gru = u_gru(compressed, state, query)
    print(f"GRU  shape={s_gru.shape}  norm={torch.norm(s_gru).item():.4f}")

    if CD_AVAILABLE:
        u_cd = FabricStateUpdater(use_cd=True)
        s_cd = u_cd(compressed, state, query, step=0, module_id='A')
        print(f"CD   shape={s_cd.shape}  norm={torch.norm(s_cd).item():.4f}")
        print(f"CD summary: {u_cd.cd_summary()}")
        print()
        # Quick goal steering test
        print("Goal steering quick test:")
        states = []
        for i in range(5):
            q_perturb = torch.randn(1, D_QUERY)
            s = u_cd(compressed, state, q_perturb, step=i, module_id='A')
            states.append(s.detach().numpy().flatten())
        from scipy.spatial.distance import pdist
        dists = pdist(states, metric='cosine')
        print(f"  Pairwise cosine dist across 5 goal vectors: mean={dists.mean():.4f}")
        print(f"  {'✓ Goal steering working' if dists.mean() > 0.01 else '❌ Still collapsing'}")