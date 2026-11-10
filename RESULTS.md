# OGI Fabric — Phase 2 Results
**Corten Research — Singleton, Dollinger & Spence**
**arXiv:2411.15832**

---

## Summary

Phase 2 replaces the GRU state update in the OGI Fabric with a CD PDE-based
update, implemented as a nonlinear elliptic solver with Lean 4 verified
convergence via `MonotoneFixedPoint.lean`.

This is not a drop-in substitution. The state update becomes a fixed-point
operator whose geometry is shaped by spatially localized viability fields,
with goal vectors modulating the field.

---

## Experimental Setup

| Parameter     | Value |
|---------------|-------|
| D_STATE       | 64    |
| D_MSG         | 64    |
| D_QUERY       | 32    |
| N_STEPS       | 40    |
| N_TRIALS      | 5     |
| Topology      | A -> B -> C -> A (feedback) |
| Middleware    | Nelson's EnsembleInspector |

CD PDE parameters (locked from validation):

| Parameter       | Value  | Notes                              |
|-----------------|--------|------------------------------------|
| L               | 1.0    | Domain length                      |
| c               | 10.0   | Saturation coefficient             |
| damping         | 0.4    | Picard iteration damping           |
| max_iter        | 10000  | Maximum iterations                 |
| tol             | 1e-9   | Convergence tolerance              |
| lambda_target   | -1.0   | Spectral calibration target        |
| kappa           | 0.2    | Care intensity (weak baseline)     |
| gamma           | 0.5    | Coherence intensity                |
| lam             | 0.1    | Contradiction cost (weak baseline) |
| sigma           | 0.12   | Gaussian bump width                |

---

## Result 1 — Attractor Convergence

**Claim:** CD PDE produces tighter attractor convergence than GRU.

| Metric         | GRU     | CD PDE  | Improvement |
|----------------|---------|---------|-------------|
| Module A var   | 0.06238 | 0.00020 | 312x        |
| Module B var   | 0.06183 | 0.00141 | 44x         |
| Module C var   | 0.06062 | 0.00064 | 95x         |
| Cosine div     | 0.4680  | 0.0023  | 204x        |

Convergence is 30-300x tighter than GRU baseline across 5 independent trials.

This aligns with the contractive behavior guaranteed by `MonotoneFixedPoint.lean`.
The proof is not decoration — it shows up in the data.

**Caveat:** 5 trials on random weights. Preliminary result.

---

## Result 2 — Scalar Beta Collapse (Key Architectural Finding)

**Finding:** Scalar `beta_b` collapses all modules to a shared attractor
regardless of goal vector. This is a structural failure, not a tuning artifact.

Confirmed via systematic beta sweep across {0.6, 0.8, 1.0, 1.2, 1.4, 1.6}:

| beta | variance | inter-module dist | goal-steering | verdict   |
|------|----------|-------------------|---------------|-----------|
| 0.6  | 0.00101  | 0.0000            | 0.0000        | COLLAPSE  |
| 0.8  | 0.00101  | 0.0000            | 0.0000        | COLLAPSE  |
| 1.0  | 0.00101  | 0.0000            | 0.0000        | COLLAPSE  |
| 1.2  | 0.00102  | 0.0000            | 0.0000        | COLLAPSE  |
| 1.4  | 0.00101  | 0.0000            | 0.0000        | COLLAPSE  |
| 1.6  | 0.00103  | 0.0001            | 0.0001        | COLLAPSE  |

**Root cause:** Scalar `beta_b` gives every module an identical viability
landscape. The goal vector is averaged away before reaching the PDE.

**Fix:** Spatially varying `beta_b` conditioned on the goal vector.
Symmetry in the control field suppresses modular differentiation.
Spatial conditioning enables it.

---

## Result 3 — Goal Steering

**Claim:** Different goal vectors produce different attractor states
from the same message.

**Test:** Fixed message seed=42, 15 independent goal vector perturbations.
Measure pairwise cosine distance between resulting module states.

**Result:** Mean pairwise cosine distance = 0.0129

All 15 solutions nontrivial and converged.

**Implementation:** Nelson's recipe (confirmed via sandbox smoke test):
1. `viability_canonical()` as baseline support field (weak: kappa=0.2)
2. `gaussian_bump_1d()` for goal-specific localized bumps
3. Spectral calibration: rescale each goal field so principal eigenvalue
   lands at lambda_target=-1.0 (same distance above threshold)
4. `a` fixed at 0.0 to isolate steering from activation strength

**Bump placement:**
- Center = 0.1 + 0.8 * median(q_norm[:16])  — first half of goal vector
- Amplitude = 1.0 + 1.0 * std(q_norm)       — goal variance drives height

---

## Theorem 2 Empirical Test

**Claim (Dollinger & Singleton, arXiv:2411.15832, Section III):**
Under Goal-Gating compression, inter-module messages converge to stable
semantic attractors determined by the receiver's goal manifold.

| Evidence | Metric                          | Result | Verdict |
|----------|---------------------------------|--------|---------|
| 1        | Cross-hop compression std       | 0.0015 | PASS    |
| 2        | Module attractor convergence    | ~0.001 | PASS    |
| 3        | SAD cosine divergence pre/post  | 0.0023 | PASS    |

---

## Architecture Convergence Note

Nelson's EnsembleInspector operates as Fabric middleware, inspecting
message content between Goal-Gate compression and state update.
This is not incidental. The Fabric protocol creates a natural inspection
point that Nelson's PatternSecurityAnalyzer / PolicyRails / Ensemble
was independently converging toward.

---

## Honest Caveats

1. All results are on random weights and synthetic data.
2. N=5 trials is a pilot. Rigorous validation requires N>=20 with
   proper statistical tests (Wilcoxon, Cohen's d).
3. Goal steering uses strong perturbations (std=0.5 goal_proj init).
   Real Gater signal magnitudes will be smaller.
4. The system demonstrates geometric structure, not task-level intelligence.
5. The path from better dynamics to better intelligence requires
   trained weights and real task evaluation.

---

## Next Steps

1. Wire the Gater to the Fabric
2. Evaluate whether geometric properties translate to controllable,
   task-relevant behavior
3. Increase N_TRIALS to 20, run proper statistical validation
4. Integrate Nelson's Renyi sidecar when available
5. Trained weight evaluation on synthetic classification task

---

## Files

| File                        | Description                              |
|-----------------------------|------------------------------------------|
| `cd_state_updater.py`       | CD PDE state updater, v4 (this commit)   |
| `ogi_fabric_standalone.ipynb` | Main Fabric notebook                   |
| `cd_v2_test.ipynb`          | Phase 2 validation experiments           |
| `phase2_beta_sweep.py`      | Beta sweep across 6 values               |
| `phase2_diversity_perturbation.py` | Attractor diversity + goal perturbation |
| `phase2_statistical_validation.py` | Statistical comparison framework  |

---

## Dependencies

- `navi-creative-determinant` (Project-Navi/navi-creative-determinant)
  Install: `pip install -e /home/sagemaker-user/navi-creative-determinant`
- `navi-SAD` (Project-Navi/navi-SAD)
  Path: `/home/sagemaker-user/navi-SAD`
- PyTorch, NumPy, SciPy, scikit-learn, matplotlib
