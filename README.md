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
from rover import run_hybrid

counts, index = run_hybrid(
    "tests/data/deterministic-interactions.xml",
    "tests/data/stochastic-gene-expression.xml",
    t_end=50.0,
    dt=1.0,
)
```

Shared store currency is molecule counts. BNGsim nM concentrations are
converted at the process boundary using SBML compartment volumes.
