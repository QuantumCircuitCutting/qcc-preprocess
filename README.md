# qcc-preprocess

Circuit preprocessing and gate decomposition

## Installation

    pip install qcc-preprocess

Or install the full suite:

    pip install quantum-circuit-cutting

## Quick Start

See [quickstart.ipynb](https://github.com/QuantumCircuitCutting/qcc-tutorials/blob/main/quickstart.ipynb)

For a local runnable example of the new wrapper API, see [examples/preprocess_api_demo.ipynb](examples/preprocess_api_demo.ipynb).
The notebook uses a small error wrapper to avoid showing full traceback/local absolute paths in output.

## Python API (QuantumCircuit Input)

You can optimize a `qiskit.QuantumCircuit` directly via the high-level wrapper API.

```python
from qiskit import QuantumCircuit
from circuit_preprocess import optimize_circuit, optimize_circuit_with_report

qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)

# Returns optimized QuantumCircuit only
qc_opt = optimize_circuit(
    qc,
    method="partitioned_lc_v2",
    processors=2,
)

# Returns detailed report (before/after metrics, runtime, equivalence check)
report = optimize_circuit_with_report(
    qc,
    method="partitioned_lc_v2",
    processors=2,
)

print(report.twoq_before, report.twoq_after)
print(report.depth_before, report.depth_after)
print(report.equivalence)
```

Available methods:

- `transpile_only`
- `lc`
- `partitioned_lc`
- `partitioned_lc_v2`

You can also use `preprocess_circuit` as an alias of `optimize_circuit`.

## Development

    git clone https://github.com/QuantumCircuitCutting/qcc-preprocess.git
    cd qcc-preprocess
    pip install -e ".[dev]"
    pytest

## License

See [LICENSE](LICENSE).
