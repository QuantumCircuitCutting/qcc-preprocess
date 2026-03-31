"""High-level preprocessing API for QuantumCircuit optimization.

`benchmark.py` と `circuit_opt.py` の機能を、
外部から使いやすい1つの入口としてまとめるモジュール。
"""

from __future__ import annotations

from typing import Literal, Sequence

from qiskit import QuantumCircuit

from .benchmark import (
	PipelineResult,
	lc_pipeline,
	partitioned_lc_pipeline,
	partitioned_lc_pipeline_v2,
	transpile_only_pipeline,
)

OptimizationMethod = Literal[
	"transpile_only",
	"lc",
	"partitioned_lc",
	"partitioned_lc_v2",
]


def available_optimization_methods() -> tuple[str, ...]:
	"""利用可能な最適化メソッド名を返す。"""
	return (
		"transpile_only",
		"lc",
		"partitioned_lc",
		"partitioned_lc_v2",
	)


def optimize_circuit_with_report(
	circuit: QuantumCircuit,
	method: OptimizationMethod = "partitioned_lc_v2",
	*,
	processors: int | None = None,
	sizes: Sequence[int] | None = None,
	optimization_level_after: int = 3,
	iters: int = 3,
	w_remote: float = 5.0,
	w_local: float = 1.0,
	max_outer_iters: int = 3,
	max_moves_per_pass: int = 50,
	target_basis: Sequence[str] = ("rz", "sx", "x", "cx"),
	force_cx: bool = True,
	exact_check_qubits_threshold: int = 10,
	name: str | None = None,
) -> PipelineResult:
	"""
	QuantumCircuit を入力として最適化パイプラインを実行し、詳細結果を返す。

	Parameters
	----------
	circuit:
		最適化対象の `qiskit.QuantumCircuit`。
	method:
		使用する最適化メソッド。
		- `"transpile_only"`
		- `"lc"`
		- `"partitioned_lc"`
		- `"partitioned_lc_v2"`

	Returns
	-------
	PipelineResult
		`benchmark.py` で定義されている詳細メトリクス付き結果。
	"""
	if not isinstance(circuit, QuantumCircuit):
		raise TypeError("circuit には qiskit.QuantumCircuit を指定してください。")

	method_name = method.lower()
	qc_in = circuit.copy()

	if method_name == "transpile_only":
		return transpile_only_pipeline(
			qc_in,
			optimization_level_after=optimization_level_after,
			name=name or "transpile_only",
		)

	if method_name == "lc":
		return lc_pipeline(
			qc_in,
			target_basis=target_basis,
			force_cx=force_cx,
			exact_check_qubits_threshold=exact_check_qubits_threshold,
			name=name or "zx_full_reduce",
		)

	if method_name == "partitioned_lc":
		return partitioned_lc_pipeline(
			qc_in,
			processors=processors,
			sizes=sizes,
			iters=iters,
			w_remote=w_remote,
			target_basis=target_basis,
			force_cx=force_cx,
			exact_check_qubits_threshold=exact_check_qubits_threshold,
			name=name or "partitioned_lc_v1",
		)

	if method_name == "partitioned_lc_v2":
		return partitioned_lc_pipeline_v2(
			qc_in,
			processors=processors,
			sizes=sizes,
			w_remote=w_remote,
			w_local=w_local,
			max_outer_iters=max_outer_iters,
			max_moves_per_pass=max_moves_per_pass,
			target_basis=target_basis,
			exact_check_qubits_threshold=exact_check_qubits_threshold,
			name=name or "partitioned_lc_v2",
		)

	methods = ", ".join(available_optimization_methods())
	raise ValueError(
		f"未知の method: {method!r}. 利用可能: {methods}"
	)


def optimize_circuit(
	circuit: QuantumCircuit,
	method: OptimizationMethod = "partitioned_lc_v2",
	**kwargs,
) -> QuantumCircuit:
	"""
	QuantumCircuit を最適化し、最適化後の回路のみを返す。

	詳細メトリクスが必要な場合は `optimize_circuit_with_report` を使用する。
	"""
	result = optimize_circuit_with_report(circuit, method=method, **kwargs)
	return result.qc_out


def optimize_circuit_auto_select(
	circuit: QuantumCircuit,
	methods: Sequence[str] | None = None,
	*,
	processors: int | None = None,
	sizes: Sequence[int] | None = None,
	optimization_level_after: int = 3,
	iters: int = 3,
	w_remote: float = 5.0,
	w_local: float = 1.0,
	max_outer_iters: int = 3,
	max_moves_per_pass: int = 50,
	target_basis: Sequence[str] = ("rz", "sx", "x", "cx"),
	force_cx: bool = True,
	exact_check_qubits_threshold: int = 10,
) -> tuple[QuantumCircuit, str, list[tuple[str, PipelineResult]]]:
	"""
	すべての最適化メソッドを実行し、2Qゲート数が最小の結果を採用する。

	Returns
	-------
	(qc_best, best_method, reports)
		- qc_best: 採用された最適化後回路
		- best_method: 採用メソッド名
		- reports: [(method, PipelineResult), ...] の一覧
	"""
	if not isinstance(circuit, QuantumCircuit):
		raise TypeError("circuit には qiskit.QuantumCircuit を指定してください。")

	method_list = tuple(methods) if methods is not None else available_optimization_methods()
	if len(method_list) == 0:
		raise ValueError("methods が空です。少なくとも1つ指定してください。")

	reports: list[tuple[str, PipelineResult]] = []
	for method in method_list:
		report = optimize_circuit_with_report(
			circuit,
			method=method,
			processors=processors,
			sizes=sizes,
			optimization_level_after=optimization_level_after,
			iters=iters,
			w_remote=w_remote,
			w_local=w_local,
			max_outer_iters=max_outer_iters,
			max_moves_per_pass=max_moves_per_pass,
			target_basis=target_basis,
			force_cx=force_cx,
			exact_check_qubits_threshold=exact_check_qubits_threshold,
		)
		reports.append((method, report))

	# Primary: twoq_after, Secondary: depth_after, Tertiary: runtime_sec
	best_method, best_report = min(
		reports,
		key=lambda mr: (mr[1].twoq_after, mr[1].depth_after, mr[1].runtime_sec),
	)
	return best_report.qc_out, best_method, reports


# 使い慣れた呼び名のエイリアス
preprocess_circuit = optimize_circuit


__all__ = [
	"OptimizationMethod",
	"available_optimization_methods",
	"optimize_circuit",
	"optimize_circuit_auto_select",
	"optimize_circuit_with_report",
	"preprocess_circuit",
]
