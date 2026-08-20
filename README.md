# Rover

**This repository is still under development.**

Statically partitioned hybrid simulator that couples:

- **BNGsim** (ODE / analytical Jacobian) on a deterministic SBML partition
- **StochMod** (constrained tau-leap) on a stochastic SBML partition

over a shared **nanomolar** array. Orchestration is a plain Python loop
(SingleCell-shaped): both modules advance from the same pre-step state, then
exchange results with **write ownership** on overlap species.

## Where one step happens

```text
HybridSimulator.run
  └─ rover.engine.run_steps
       ├─ s0 = counts          # shared nM
       ├─ stoch.advance_from(s0, dt)   # StochModModule (nM ↔ molecules; TFs as nM)
       ├─ bng.advance_from(s0, dt)     # BngsimModule (identity nM; mRNA frozen)
       └─ exchange(s0, s_stoch, s_bng) # owner-takes-all on overlap
```

Edit coupling in `[rover/engine.py](rover/engine.py)` (`run_steps` / `exchange_counts`)
or `advance_from` under `[rover/modules/](rover/modules/)`.

Partitioning is **file membership** plus stoichiometric **write ownership**:
overlap mRNA/genes are stoch-owned; overlap TFs (modifiers only in StochMod) are
det-owned. Exchange takes the owner’s post-step value (not `Δ_stoch + Δ_bng`).
Det-owned TFs are passed into StochMod as nM (SingleCell-like), not molecule counts.

## Quickstart

[StochMod](https://pypi.org/project/stochmod/) and [BNGsim](https://pypi.org/project/bngsim/)
install from PyPI with Rover:

```bash
pip install -e ".[dev]"

python basic_hybrid.py
pytest
```

StochMod compiles per-model C propensity code on first load of each SBML, so a
C compiler (`gcc`, `clang`, or MSVC `cl`) must be available. Wheels from PyPI
already include the Python extension itself.

For local StochMod development, install an editable clone first
(`pip install -e /path/to/StochMod`) so it shadows the PyPI package, then
install Rover as above.

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


| Piece          | Where                                               | Why                                  |
| -------------- | --------------------------------------------------- | ------------------------------------ |
| Live `_counts` | 1-D `float64` nM in RAM                             | Hot path for coupling                |
| Trajectory     | `(n_points, n_species)` memory **or** memmap `.npy` | History only; one row write per step |


Do **not** have BNGsim/StochMod dump to disk each step — that would dominate cost.
The orchestrator copies the live vector into the next trajectory row (sequential
write; memmap is fine because the OS caches pages).

## Performance notes

**Absolute time is orchestrator-owned.** Trajectory times are `t0 + step*dt`.
Each coupling step advances a local interval `[0, dt]` in BNGsim (interactive
clock rewound before `advance(dt)`). StochMod leaps once by `dt` per call.

Hybrid coupling is **not** the same cost model as a standalone batch run:


| Mode                                                   | What happens                                                |
| ------------------------------------------------------ | ----------------------------------------------------------- |
| `StochMod.run(0, T, dt)` / `bngsim.Simulator.run(...)` | One Python call; entire trajectory stays in C               |
| `sim.run(t_end=T)` with coupling `dt`                  | `T/dt` Python↔C round-trips (set_state / advance / convert) |


So `t_end=259200, dt=1` means **259 200 coupling steps**. Prefer a larger
coupling `dt` when the biology allows it (e.g. `dt=30` → ~8640 steps for 3 days).

BNGsim defaults: codegen on, `jacobian="analytical"`, `rtol=1e-4`, `atol=1e-6`,
`max_steps=1e8`. If codegen fails, Rover falls back to the interpreted RHS; if
analytical Jacobian construction fails, it falls back to finite differences.
On some Windows consoles codegen can fail with a `charmap` encoding error.

## Preparing your own SBML pair

Rover does not ingest a single model. You split **your** network into two
SBML files and pass them as:

```python
HybridSimulator(deterministic_sbml, stochastic_sbml, dt=...)
```

The bundled examples happen to be gene expression plus protein binding
(mRNA, genes, TFs). That biology is **not required**. Any species can live on either side of the split. Rover never classifies a species by what it represents; only by which file lists it and whether it is a reactant/product there.

### What each file contains


| File               | Simulator           | Put here                                       |
| ------------------ | ------------------- | ---------------------------------------------- |
| Deterministic SBML | BNGsim (ODE)        | Reactions you want integrated continuously     |
| Stochastic SBML    | StochMod (tau-leap) | Reactions you want as discrete molecule events |


A species may appear in **one file or both**. IDs are the join key: the same
`species id` in both files is one shared state entry. Exclusive species
(present in only one file) are fine.

Typical pattern, independent of molecule type:

1. Put a reaction in **exactly one** file — the side that should fire it.
2. If the other side’s rate laws need that species as an input, also declare
  it in the other file, usually as a **modifier** (not a reactant/product).
3. Keep compartment `id`s and sizes consistent for shared species.

Do **not** put the same reaction in both files. Species that are
reactants/products in **both** files are ambiguous (see ownership below);
avoid that unless you intend the name-based fallback.

### Write ownership (who updates shared species)

After each coupling step, overlap species take the **owner’s** post-step
value. Ownership is inferred from stoichiometry, not from names like
`mrna` / `gene` / `prot`:

- **Reactant or product only in the stochastic file** → StochMod owns it.
BNGsim holds it fixed over the ODE window (so the deterministic file can
still *read* it, e.g. as a modifier).
- **Reactant or product only in the deterministic file** → BNGsim owns it.
StochMod sees it as nanomolar (not molecule counts) and does not write it
back — the usual pattern for a regulator that appears only as a modifier
in stochastic rate laws.
- **Modifier-only** in a file does **not** confer ownership.
- If a species is a reactant/product in **both** files, Rover falls back to
a name heuristic (`mrna` / `gene` → stochastic, otherwise deterministic).
Do not rely on that for a custom model: keep stoichiometric participation
on one side only.

So: a low-copy ion channel, a promoter, or an enzyme can be stochastic;
a high-abundance metabolite or a transcription factor can be deterministic.
Put the species as reactant/product in the file that should own it, and as
a modifier (if needed) in the other.

### Units and ICs

Shared state is **nanomolar**. Use SBML `initialConcentration` in nM,
compartment sizes in litres, and substance units of nanomole (the example
files define `substance` as `mole` with `scale="-9"`). StochMod converts
stoch-owned species to molecule counts with `nM · V · N_A · 1e-9`;
det-owned overlap is passed through as nM, so stochastic kinetic laws that
read those species should be written in nM, not counts.

If both files set an overlap species’ initial condition, BNGsim’s value is
used and a mismatch is logged.

Example layout (bundled LR pair):
`[tests/data/LR/deterministic-interactions.xml](tests/data/LR/deterministic-interactions.xml)`
and
`[tests/data/LR/stochastic-gene-expression.xml](tests/data/LR/stochastic-gene-expression.xml)`.