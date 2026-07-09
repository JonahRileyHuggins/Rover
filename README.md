# Rover

Statically partitioned hybrid simulator that couples:

- **BNGsim** (ODE / Jacobian) on a deterministic SBML partition
- **StochMod** (constrained tau-leap) on a stochastic SBML partition

over a shared **molecule-count** array store, orchestrated with
[process-bigraph](https://github.com/vivarium-collective/process-bigraph).

## Quickstart

```bash
# Install StochMod (editable) then Rover
pip install -e ../StochMod
pip install -e ".[dev]"

python basic_bigraph.py
pytest
```

```python
from rover import HybridSimulator

sim = HybridSimulator(
    "tests/data/deterministic-interactions.xml",
    "tests/data/stochastic-gene-expression.xml",
    dt=1.0,
)

# Species and/or parameters by id (shared store + routed to the owning kernel)
sim.update("cyt_mrna__LIGAND_", 10)
sim.update({"kTL1_1": 2.0, "kTC1_1": 0.01})

counts = sim.run(t_end=60.0)
df = sim.to_dataframe()          # one-row pandas frame, columns = species ids
sim.reset()                      # restore ICs + parameters
```

One-shot helper (rebuilds models each call — prefer `HybridSimulator` for sweeps):

```python
from rover import run_hybrid
counts, index = run_hybrid(det_xml, stoch_xml, t_end=50.0, dt=1.0)
```

Shared store currency is molecule counts. BNGsim nM concentrations are
converted at the process boundary using SBML compartment volumes.

## Performance notes

`run` / `run_hybrid` default to `engine="split"`: a tight Python loop that calls
each simulator's `apply_inplace` once per coupling step. That is much faster
than `engine="composite"` (full process-bigraph scheduling every step).

Still, hybrid coupling is **not** the same cost model as a standalone batch run:

| Mode | What happens |
|------|----------------|
| `StochMod.run(0, T, dt)` / `bngsim.Simulator.run(...)` | One Python call; entire trajectory stays in C |
| `sim.run(t_end=T)` with coupling `dt` | `T/dt` Python↔C round-trips (set_state / advance / convert) |

So `t_end=259200, dt=1` means **259 200 coupling steps**. Prefer a larger
coupling `dt` when the biology allows it (e.g. `dt=30` → ~8640 steps for 3 days).

BNGsim INFO logs report whether codegen succeeded. On some Windows consoles
codegen can fail with a `charmap` encoding error; Rover falls back to the
interpreted RHS and logs a warning.
