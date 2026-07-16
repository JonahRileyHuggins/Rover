# Rover

Statically partitioned hybrid simulator that couples:

- **BNGsim** (ODE / analytical Jacobian) on a deterministic SBML partition
- **StochMod** (constrained tau-leap) on a stochastic SBML partition

over a shared **nanomolar** array. Orchestration is a plain Python loop
(SingleCell-shaped): both modules advance from the same pre-step state, then
exchange results (exclusive copy + additive overlap deltas).

## Where one step happens

```text
HybridSimulator.run
  └─ rover.engine.run_steps
       ├─ s0 = counts          # shared nM
       ├─ stoch.advance_from(s0, dt)   # StochModModule (nM ↔ molecules)
       ├─ bng.advance_from(s0, dt)     # BngsimModule (identity nM; clock → 0)
       └─ exchange(s0, s_stoch, s_bng) # membership merge
```

Edit coupling in [`rover/engine.py`](rover/engine.py) (`run_steps` / `exchange_counts`)
or `advance_from` under [`rover/modules/`](rover/modules/).

Partitioning is **file membership** only (no annotations / species-name rules).
Overlap species are exchange variables: each module may read them; after both
steps, shared values get `s0 + Δ_stoch + Δ_bng` so a modifier-only module
(`Δ ≈ 0`) does not wipe the producer’s update.

## Quickstart

```bash
# Install StochMod (editable) then Rover
pip install -e ../StochMod
pip install -e ".[dev]"

python basic_hybrid.py
pytest
```

```python
from rover import HybridSimulator

sim = HybridSimulator(
    "tests/data/LR/deterministic-interactions.xml",
    "tests/data/LR/stochastic-gene-expression.xml",
    dt=1.0,
)

sim.update("cyt_mrna__LIGAND_", 0.001582)
sim.update({"kTL1_1": 2.0, "kTC1_1": 0.01})

traj = sim.run(t_end=60.0)           # shape (61, n_species); live state stays 1-D nM
df = sim.to_dataframe()              # columns: time + species ids (nM)
# Large runs — pre-size a memmap .npy (OS page cache; kernels never open the file):
# traj = sim.run(t_end=259200, dt=30, results_path="out/traj.npy")
sim.reset()
```

One-shot helper (rebuilds models each call — prefer `HybridSimulator` for sweeps):

```python
from rover import run_hybrid
traj, index = run_hybrid(det_xml, stoch_xml, t_end=50.0, dt=1.0)
```

Shared store currency is nanomolar. BNGsim storage is already nM (identity
bridge). StochMod converts to molecule counts at the module boundary using SBML
compartment volumes (`nM · V · N_A · 1e-9`), matching SingleCell.

### Trajectory storage

| Piece | Where | Why |
|-------|--------|-----|
| Live `_counts` | 1-D `float64` nM in RAM | Hot path for coupling |
| Trajectory | `(n_points, n_species)` memory **or** memmap `.npy` | History only; one row write per step |

Do **not** have BNGsim/StochMod dump to disk each step — that would dominate cost.
The orchestrator copies the live vector into the next trajectory row (sequential
write; memmap is fine because the OS caches pages).

## Performance notes

**Absolute time is orchestrator-owned.** Trajectory times are `t0 + step*dt`.
Each coupling step advances a local interval `[0, dt]` in BNGsim (interactive
clock rewound before `advance(dt)`). StochMod leaps once by `dt` per call.

Hybrid coupling is **not** the same cost model as a standalone batch run:

| Mode | What happens |
|------|----------------|
| `StochMod.run(0, T, dt)` / `bngsim.Simulator.run(...)` | One Python call; entire trajectory stays in C |
| `sim.run(t_end=T)` with coupling `dt` | `T/dt` Python↔C round-trips (set_state / advance / convert) |

So `t_end=259200, dt=1` means **259 200 coupling steps**. Prefer a larger
coupling `dt` when the biology allows it (e.g. `dt=30` → ~8640 steps for 3 days).

BNGsim defaults: codegen on, `jacobian="analytical"`, `rtol=1e-4`, `atol=1e-6`,
`max_steps=1e8`. If codegen fails, Rover falls back to the interpreted RHS; if
analytical Jacobian construction fails, it falls back to finite differences.
On some Windows consoles codegen can fail with a `charmap` encoding error.
