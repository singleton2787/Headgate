# Headgate

Private implementation repository for the OGI Executive Gater and Fabric architecture.  
**IconSoftLLC** — Singleton, Dollinger, Spence

This is a working document. It tracks what's proven, what's blocked, and what comes next. If you're reading this you're either Dan or Nelson — orient here first before diving into code.

The public-facing math is at [ogi-math-foundations](https://github.com/singleton2787/ogi-math-foundations). That repo is the academic front door. This is where the machine lives.

---

## Current State (April 2026)

### What's Done

**Executive Gater** — solid. Not a work in progress.

- Routing geometry validated across three seeds (42, 123, 777). Result figures in `experiments/results/`.
- navi-SAD integrated as impartial observer. SAD wire confirmed clean — see `instruments/ogi_navi_sad/`.
- Theorem 1 (Lipschitz stability) proved and empirically verified. The Gater's attention-based weighting is stable under noise up to σ = 1.0.
- Top-K routing reduces complexity from O(n²dₘ) to O(k²dₘ + n(d꜀d_a)) — proved, not just claimed.
- The Gater does not need more testing. It needs integration with the Fabric, and that integration is blocked downstream, not here.

**Fabric** — GRU baseline is done. `ogi_fabric_standalone.ipynb` in `fabric/`.

- Inter-module message passing works at the GRU baseline level.
- The coherence objective is implemented. +73% compute overhead measured (the Coherence Tax).
- What it doesn't have yet: goal-differentiated module states. That's the CD PDE problem.

---

## The Dependency Chain

```
Gater (done)
    ↓
Fabric — GRU baseline (done)
    ↓
CD PDE goal steering (BLOCKED — waiting on Nelson / fields.py)
    ↓
Gater + Fabric integration (blocked by above)
```

The integration is not blocked by anything on the Gater or Fabric side. It's blocked because there's nothing observable to measure until goal steering is working. The whole point of wiring the Gater to the Fabric is that routing decisions propagate through goal-differentiated module states. If goal steering is zero, the integration produces nothing meaningful.

**Do not wire Gater to Fabric until goal steering is confirmed.**

---

## What's Blocked

**CD PDE goal steering** — waiting on Nelson's response on `fields.py`.

The beta sweep in `experiments/` was inconclusive because of a convergence problem in the field construction. The sweep itself is fast once the field construction is right. This is not a hard problem — it's a waiting problem.

Once Nelson confirms the correct field construction:
1. Re-run the beta sweep with converged solutions
2. Confirm goal steering is working across seeds
3. Wire Gater to Fabric
4. Re-run SAD observer — expect attractor geometry to shift

---

## What To Do While Waiting on Nelson

- Review `cd_state_updater.py` in `gater/` — this is the control dynamics state updater. It's close but tied to the convergence problem. Worth a read.
- The SAD export data in `experiments/` (`ogi_navisad_export.json`, `takens_bridge_results.json`) has not been fully analyzed. Per-head PE on first-differenced trajectories shows structural signal — 338/1024 heads with |d| > 0.5 in 3+ mode/segment combos, 4.6:1 directional asymmetry. That deserves a closer look independent of the integration work.
- `implementation/` is empty. That's intentional — the implementation paper (step 3 of the sequence) starts here when we're ready.

---

## Repository Structure

```
headgate/
├── gater/
│   ├── ogi_executive_gater.py       # Primary Gater implementation
│   ├── ogi_executive_gater_v3.py    # v3 (reference)
│   └── cd_state_updater.py          # Control dynamics state updater
├── fabric/
│   └── ogi_fabric_standalone.ipynb  # Fabric experiment — GRU baseline
├── instruments/
│   ├── oginavisadv3.py              # navi-SAD v3
│   ├── og_navisad_v3.py             # navi-SAD integration (Gater observer)
│   └── ogi_navi_sad/                # Full SAD instrument folder
│       ├── ordinal.py               # Bandt-Pompe ordinal patterns
│       ├── pe_features.py           # Per-(layer,head) PE extraction
│       ├── ogi_sad_integrated.py    # Integrated SAD runner
│       └── ...
├── experiments/
│   ├── ogi_navisad_export.json      # SAD output — full
│   ├── ogi_navisad_export_sample.json
│   ├── takens_bridge_results.json   # Takens embedding validation
│   └── results/
│       ├── ogi_summary_results.png  # v3 summary chart
│       ├── fabric_attractor_geometry.png
│       ├── fabric_gate_selectivity.png
│       └── fabric_sad_wire.png
├── implementation/                  # Empty — step 3 paper starts here
├── tests/                           # Empty — test harness goes here
└── docs/
    └── README.txt                   # Original v3 notes
```

---

## Decisions Log

**Why private?**
The Gater and Fabric are potentially licensable IP under IconSoftLLC. The math foundations are public and citable. The implementation stays controlled until we decide otherwise. This can transfer to a shared org when Dan and Nelson are formally in.

**Why monorepo?**
We're moving fast and the components are tightly coupled right now. Split into separate repos if/when Headgate grows past what one repo can hold cleanly.

**Why no license?**
All rights reserved until we decide on a commercialization path. Apache 2.0 or dual license (open/commercial) are the likely candidates when we're ready. Do not add a license without a conversation.

**navi-SAD as impartial observer**
Nelson's instrument sits in `instruments/` here and is referenced (not duplicated) in the public repo. The relationship is intentional — SAD is the observer, Headgate is the system under observation. When Nelson formalizes the collaboration, the instrument integration belongs in whatever shared org we land on.

**The beta sweep**
The inconclusive result was not a failure of the approach — it was a field construction problem. Do not interpret the sweep results as evidence against the CD PDE framing. The convergence issue is known and has a fix pending.

---

## Collaborators

**Michael Singleton** — Iconsoft LLC. Gater architecture, math foundations, integration.  
**Daniel Dollinger** — OGI framework co-author. [arXiv:2411.15832](https://arxiv.org/abs/2411.15832)  
**Nelson Spence** — Project-Navi. navi-SAD instrument, CD PDE formalization, fields.py.

Questions: [mike@mikesingleton.info](mailto:mike@mikesingleton.info)

---

## Related

- **Public math repo:** [ogi-math-foundations](https://github.com/singleton2787/ogi-math-foundations)
- **navi-SAD:** [Project-Navi/navi-SAD](https://github.com/Project-Navi/navi-SAD)
- **OGI Framework paper:** [arXiv:2411.15832](https://arxiv.org/abs/2411.15832)
- **Math foundations paper:** [Zenodo DOI: 10.5281/zenodo.19135205](https://doi.org/10.5281/zenodo.19135205)
