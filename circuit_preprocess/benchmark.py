"""
benchmark.py

- ランダム回路生成
- パイプライン用の薄いラッパ
- ベンチマーク実行
- ベンチマーク結果の可視化
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile

from .circuit_opt import (
    count_two_qubit_gates,
    cz_to_cx,
    fast_equiv_probe,
    remote_aware_zx_reduce,
    simplify_single_qubit,
    unitary_equiv,
    zx_reduce_circuit,
    build_interaction_graph,
    count_remote_twoq,
    kway_partition,
    eff_cost,
    zx_local_search_min_cost,
    zx_local_search_min_cost_with_cancel
)


# ============================================================
# ランダム回路生成 & 基底分解
# ============================================================

def random_cx_circuit(
    num_qubits: int,
    depth: int,
    cx_density: float,
    seed: Optional[int] = None,
) -> QuantumCircuit:
    """
    CXゲートの密度・量子ビット数・回路深さを指定して
    ランダム量子回路を生成する。
    """
    if not (0.0 <= cx_density <= 1.0):
        raise ValueError("cx_density は 0.0〜1.0 の範囲で指定してください。")
    if num_qubits <= 0:
        raise ValueError("num_qubits は 1 以上を指定してください。")
    if depth <= 0:
        raise ValueError("depth は 1 以上を指定してください。")

    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(num_qubits)

    one_qubit_gates = ["rx", "ry", "rz", "h", "s", "t"]

    for _ in range(depth):
        # このレイヤーでまだ使われていない量子ビットのリスト
        available = list(range(num_qubits))
        rng.shuffle(available)

        i = 0
        while i < len(available):
            q = available[i]

            # 残りが2つ以上あって、確率的にCXを入れるとき
            if (len(available) - i >= 2) and (rng.random() < cx_density):
                # パートナーとなる別の量子ビットをランダムに選ぶ
                partner_idx = rng.integers(i + 1, len(available))
                q2 = available[partner_idx]

                # q2 を i+1 の位置に持ってきておくとわかりやすい
                available[i + 1], available[partner_idx] = (
                    available[partner_idx],
                    available[i + 1],
                )

                # CXゲートを追加（q を制御、q2 を標的に）
                qc.cx(q, q2)
                i += 2  # 2つの量子ビットを消費
            else:
                # 1量子ビットゲートをランダムに適用
                gate = rng.choice(one_qubit_gates)
                if gate in ["rx", "ry", "rz"]:
                    theta = 2 * np.pi * rng.random()
                    getattr(qc, gate)(theta, q)
                else:
                    getattr(qc, gate)(q)
                i += 1  # 1つの量子ビットを消費

    return qc


def decompose_to_basis(
    circuit: QuantumCircuit,
    basis_gates: Sequence[str] = ("cx", "rx", "ry", "rz"),
    optimization_level: int = 0,
) -> QuantumCircuit:
    """
    与えられた量子回路を指定したゲートセットに分解する。
    """
    basis = list(basis_gates)

    # トランスパイラが内部で使うことがある id を念のため追加
    if "id" not in basis:
        basis.append("id")

    new_circuit = transpile(
        circuit,
        basis_gates=basis,
        optimization_level=optimization_level,
    )
    return new_circuit


# ============================================================
# パイプライン薄ラッパ
# ============================================================

@dataclass
class PipelineResult:
    name: str
    qc_in: QuantumCircuit
    qc_out: QuantumCircuit
    twoq_before: int
    twoq_after: int
    depth_before: int
    depth_after: int
    remote_before: Optional[int] = None
    remote_after: Optional[int] = None
    equivalence: Optional[str] = None
    runtime_sec: float = 0.0
    extra: Dict[str, float] | None = None


def transpile_only_pipeline(
    qc_in: QuantumCircuit,
    optimization_level_after: int = 3,
    name: str = "transpile_only",
) -> PipelineResult:
    """
    ベースライン: transpile のみで 2Q / remote をどこまで削れるか。
    - before: optimization_level=0 の雑なトランスパイル結果
    - after : optimization_level_after のトランスパイル結果
    """
    start = time.perf_counter()

    base = transpile(qc_in, optimization_level=0)
    before_twoq = count_two_qubit_gates(base)
    depth_before = base.depth()

    reduced = transpile(base, optimization_level=optimization_level_after)
    after_twoq = count_two_qubit_gates(reduced)
    depth_after = reduced.depth()

    # 等価性チェック
    n = base.num_qubits
    if n <= 10:
        ok, dist = unitary_equiv(base, reduced)
        verdict = f"exact up to global phase: {ok} (Frobenius distance={dist:.2e})"
    else:
        ok = fast_equiv_probe(base, reduced)
        verdict = f"probabilistic check passed: {ok} (shots=64)"

    runtime = time.perf_counter() - start

    return PipelineResult(
        name=name,
        qc_in=base,
        qc_out=reduced,
        twoq_before=before_twoq,
        twoq_after=after_twoq,
        depth_before=depth_before,
        depth_after=depth_after,
        equivalence=verdict,
        runtime_sec=runtime,
    )


def lc_pipeline(
    qc_in: QuantumCircuit,
    target_basis: Sequence[str] = ("rz", "sx", "x", "cx"),
    force_cx: bool = True,
    exact_check_qubits_threshold: int = 10,
    name: str = "zx_full_reduce",
) -> PipelineResult:
    """
    既存の ZX full_reduce ベース簡約を
    「ベンチマーク用の1本のパイプライン」としてラップしたもの。
    """
    start = time.perf_counter()

    base = transpile(qc_in, optimization_level=0)
    before_twoq = count_two_qubit_gates(base)
    depth_before = base.depth()

    # ZX 簡約
    reduced = zx_reduce_circuit(base)

    # CZ → CX に変換する場合
    if force_cx:
        reduced = cz_to_cx(reduced)

    # 単一量子ビット整理 & target basis 化
    reduced = simplify_single_qubit(reduced, basis=target_basis)

    after_twoq = count_two_qubit_gates(reduced)
    depth_after = reduced.depth()

    # 等価性チェック
    n = base.num_qubits
    if n <= exact_check_qubits_threshold:
        ok, dist = unitary_equiv(base, reduced)
        verdict = f"exact up to global phase: {ok} (Frobenius distance={dist:.2e})"
    else:
        ok = fast_equiv_probe(base, reduced)
        verdict = f"probabilistic check passed: {ok} (shots=64)"

    runtime = time.perf_counter() - start

    return PipelineResult(
        name=name,
        qc_in=base,
        qc_out=reduced,
        twoq_before=before_twoq,
        twoq_after=after_twoq,
        depth_before=depth_before,
        depth_after=depth_after,
        equivalence=verdict,
        runtime_sec=runtime,
    )


def partitioned_lc_pipeline(
    qc_in: QuantumCircuit,
    processors: Optional[int] = None,
    sizes: Optional[Sequence[int]] = None,
    iters: int = 3,
    w_remote: float = 2.0,
    target_basis: Sequence[str] = ("rz", "sx", "x", "cx"),
    force_cx: bool = True,
    exact_check_qubits_threshold: int = 10,
    name: str = "partitioned_lc_v1",
) -> PipelineResult:
    """
    「プロセッサ分割 + リモート志向 LC 簡約」（旧版）を 1 本のパイプラインとして実装。
    remote_aware_zx_reduce を利用。
    """
    start = time.perf_counter()

    base = transpile(qc_in, optimization_level=0)
    n = base.num_qubits

    # sizes が指定されていなければ均等割り
    if sizes is None:
        assert processors is not None, "processors か sizes のどちらかは指定してください。"
        base_size = n // processors
        rem = n % processors
        sizes = [base_size + (1 if i < rem else 0) for i in range(processors)]
    else:
        sizes = list(sizes)
        assert sum(sizes) == n, "sum(sizes) が num_qubits と一致している必要があります。"

    # 1回目のパーティション
    G = build_interaction_graph(base)
    part_of = kway_partition(G, sizes)

    remote_before, local_before = count_remote_twoq(base, part_of)
    before_twoq = count_two_qubit_gates(base)
    depth_before = base.depth()

    best_qc = base
    best_remote = remote_before

    # 簡単な outer-iteration
    for _ in range(iters):
        candidate = remote_aware_zx_reduce(best_qc, part_of, w_remote=w_remote)

        # パーティションは一旦固定したまま remote count を比較
        cand_remote, _ = count_remote_twoq(candidate, part_of)
        if cand_remote <= best_remote:
            best_qc = candidate
            best_remote = cand_remote

        # candidate に対して再パーティションし直してもよい
        G = build_interaction_graph(best_qc)
        part_of = kway_partition(G, sizes)

    # 最後に CZ→CX & basis 化を合わせる
    reduced = best_qc
    if force_cx:
        reduced = cz_to_cx(reduced)
    reduced = simplify_single_qubit(reduced, basis=target_basis)

    remote_after, local_after = count_remote_twoq(reduced, part_of)
    after_twoq = count_two_qubit_gates(reduced)
    depth_after = reduced.depth()

    # 等価性チェック
    if n <= exact_check_qubits_threshold:
        ok, dist = unitary_equiv(base, reduced)
        verdict = f"exact up to global phase: {ok} (Frobenius distance={dist:.2e})"
    else:
        ok = fast_equiv_probe(base, reduced)
        verdict = f"probabilistic check passed: {ok} (shots=64)"

    runtime = time.perf_counter() - start

    return PipelineResult(
        name=name,
        qc_in=base,
        qc_out=reduced,
        twoq_before=before_twoq,
        twoq_after=after_twoq,
        depth_before=depth_before,
        depth_after=depth_after,
        remote_before=remote_before,
        remote_after=remote_after,
        equivalence=verdict,
        runtime_sec=runtime,
        extra={
            "local_before": float(local_before),
            "local_after": float(local_after),
        },
    )


def partitioned_lc_pipeline_v2(
    qc_in: QuantumCircuit,
    processors: Optional[int] = None,
    sizes: Optional[Sequence[int]] = None,
    w_remote: float = 5.0,
    w_local: float = 1.0,
    max_outer_iters: int = 3,
    max_moves_per_pass: int = 50,
    target_basis: Sequence[str] = ("rz", "sx", "x", "cx"),
    exact_check_qubits_threshold: int = 10,
    name: str = "partitioned_lc_v2",
) -> PipelineResult:
    """
    ZX ローカルサーチ + rollback 付きの新しい partition-aware パイプライン。

    - スタート地点は「optimization_level=3 の transpile 結果」
    - zx_local_search_min_cost は eff_cost が下がる move だけ採用する
      → 初期 transpile 案より eff_cost が悪化しない設計（理想）
    """
    start = time.perf_counter()

    base = transpile(qc_in, optimization_level=3)
    n = base.num_qubits

    # sizes が指定されていなければ均等割り
    if sizes is None:
        assert processors is not None, "processors か sizes のどちらかは指定してください。"
        base_size = n // processors
        rem = n % processors
        sizes = [base_size + (1 if i < rem else 0) for i in range(processors)]
    else:
        sizes = list(sizes)
        assert sum(sizes) == n, "sum(sizes) が num_qubits と一致している必要があります。"

    # 初期 eff_cost を評価
    G0 = build_interaction_graph(base)
    part0 = kway_partition(G0, sizes)
    cost0, remote0, local0 = eff_cost(base, part0, w_remote=w_remote, w_local=w_local)

    before_twoq = count_two_qubit_gates(base)
    depth_before = base.depth()

    # ZX ローカルサーチ
    # zx_qc, part_final, stats = zx_local_search_min_cost(
    #     base,
    #     sizes=list(sizes),
    #     w_remote=w_remote,
    #     w_local=w_local,
    #     max_outer_iters=max_outer_iters,
    #     max_moves_per_pass=max_moves_per_pass,
    # )

    zx_qc, part_final, stats = zx_local_search_min_cost_with_cancel(
    base,
    sizes=list(sizes),
    w_remote=w_remote,
    w_local=w_local,
    max_outer_iters=max_outer_iters,
    max_moves_per_pass=max_moves_per_pass,
    cancel_each_trial=True,
    )

    # 最後に CZ→CX & basis 化
    reduced = zx_qc
    reduced = cz_to_cx(reduced)
    reduced = simplify_single_qubit(reduced, basis=target_basis)

    # 最終 eff_cost
    Gf = build_interaction_graph(reduced)
    part_f = kway_partition(Gf, sizes)
    cost_f, remote_f, local_f = eff_cost(reduced, part_f, w_remote=w_remote, w_local=w_local)

    after_twoq = count_two_qubit_gates(reduced)
    depth_after = reduced.depth()

    # 等価性チェック（初期の base との比較）
    if n <= exact_check_qubits_threshold:
        ok, dist = unitary_equiv(base, reduced)
        verdict = f"exact up to global phase: {ok} (Frobenius distance={dist:.2e})"
    else:
        ok = fast_equiv_probe(base, reduced)
        verdict = f"probabilistic check passed: {ok} (shots=64)"

    runtime = time.perf_counter() - start

    extra: Dict[str, float] = {
        "eff_cost_initial": float(cost0),
        "eff_cost_final": float(cost_f),
        "remote_initial": float(remote0),
        "remote_final": float(remote_f),
        "local_initial": float(local0),
        "local_final": float(local_f),
    }
    extra.update({k: float(v) for k, v in stats.items()})

    return PipelineResult(
        name=name,
        qc_in=base,
        qc_out=reduced,
        twoq_before=before_twoq,
        twoq_after=after_twoq,
        depth_before=depth_before,
        depth_after=depth_after,
        remote_before=remote0,
        remote_after=remote_f,
        equivalence=verdict,
        runtime_sec=runtime,
        extra=extra,
    )


# ============================================================
# ベンチマーク・ループ
# ============================================================

@dataclass
class BenchmarkSample:
    pipeline: str
    num_qubits: int
    depth: int
    cx_density: float
    seed: int
    twoq_before: int
    twoq_after: int
    depth_before: int
    depth_after: int
    runtime_sec: float
    remote_before: Optional[int] = None
    remote_after: Optional[int] = None
    eff_cost_before: Optional[float] = None
    eff_cost_after: Optional[float] = None


def benchmark_pipeline_on_random(
    pipeline_fn: Callable[[QuantumCircuit], PipelineResult],
    num_qubits_list: Sequence[int],
    depth_list: Sequence[int],
    cx_density_list: Sequence[float],
    shots_per_setting: int = 3,
    seed_offset: int = 0,
    processors_for_eval: Optional[int] = None,
    remote_weight: float = 5.0,
    local_weight: float = 1.0,
) -> List[BenchmarkSample]:
    """
    与えられた pipeline_fn を、ランダム回路のグリッドサーチ上で評価する。
    remote / eff_cost も必要なら一緒に記録する。
    """
    results: List[BenchmarkSample] = []
    base_seed = seed_offset

    for nq in num_qubits_list:
        for d in depth_list:
            for dens in cx_density_list:
                for s in range(shots_per_setting):
                    seed = base_seed + s
                    qc_rand = random_cx_circuit(
                        num_qubits=nq,
                        depth=d,
                        cx_density=dens,
                        seed=seed,
                    )
                    qc_rand = decompose_to_basis(qc_rand)

                    res = pipeline_fn(qc_rand)

                    # remote / eff_cost 評価
                    remote_before = None
                    remote_after = None
                    eff_before = None
                    eff_after = None

                    if processors_for_eval is not None and processors_for_eval > 1:
                        base_circ = res.qc_in
                        n_eval = base_circ.num_qubits
                        p_eval = min(processors_for_eval, n_eval)
                        if p_eval < 1:
                            p_eval = 1

                        base_size = n_eval // p_eval
                        rem_q = n_eval % p_eval
                        sizes = [
                            base_size + (1 if i < rem_q else 0)
                            for i in range(p_eval)
                        ]

                        # before
                        G_before = build_interaction_graph(base_circ)
                        part_before = kway_partition(G_before, sizes)
                        cost_b, remote_before, local_before = eff_cost(
                            base_circ,
                            part_before,
                            w_remote=remote_weight,
                            w_local=local_weight,
                        )

                        # after
                        G_after = build_interaction_graph(res.qc_out)
                        part_after = kway_partition(G_after, sizes)
                        cost_a, remote_after, local_after = eff_cost(
                            res.qc_out,
                            part_after,
                            w_remote=remote_weight,
                            w_local=local_weight,
                        )

                        eff_before = cost_b
                        eff_after = cost_a

                    results.append(
                        BenchmarkSample(
                            pipeline=res.name,
                            num_qubits=nq,
                            depth=d,
                            cx_density=dens,
                            seed=seed,
                            twoq_before=res.twoq_before,
                            twoq_after=res.twoq_after,
                            depth_before=res.depth_before,
                            depth_after=res.depth_after,
                            runtime_sec=res.runtime_sec,
                            remote_before=remote_before,
                            remote_after=remote_after,
                            eff_cost_before=eff_before,
                            eff_cost_after=eff_after,
                        )
                    )

    return results


# ============================================================
# 可視化・集計ユーティリティ
# ============================================================

def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.array(list(values), dtype=float)


def plot_twoq_reduction_hist(
    samples: List[BenchmarkSample],
    title: str = "Two-qubit gate reduction ratio (after / before)",
) -> None:
    """
    2量子ビットゲートの削減率 after/before のヒストグラム。
    複数 pipeline が混ざっているなら pipeline ごとに色を変える。
    """
    plt.figure()
    by_pipe: Dict[str, List[float]] = {}
    for s in samples:
        if s.twoq_before == 0:
            continue
        ratio = s.twoq_after / s.twoq_before
        by_pipe.setdefault(s.pipeline, []).append(ratio)

    for name, ratios in by_pipe.items():
        arr = _as_array(ratios)
        plt.hist(arr, bins=20, alpha=0.5, label=name)

    plt.xlabel("twoq_after / twoq_before")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_runtime_vs_size(
    samples: List[BenchmarkSample],
    title: str = "Runtime vs number of qubits",
) -> None:
    """
    キュビット数に対する実行時間の散布図。
    depth や density が混ざっていてもざっくりスケーリングを見る用。
    """
    plt.figure()
    pipelines = sorted(set(s.pipeline for s in samples))

    for p in pipelines:
        xs = [s.num_qubits for s in samples if s.pipeline == p]
        ys = [s.runtime_sec for s in samples if s.pipeline == p]
        plt.scatter(xs, ys, label=p)

    plt.xlabel("num_qubits")
    plt.ylabel("runtime [sec]")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_remote_ratio(
    samples: List[BenchmarkSample],
    title: str = "Remote 2Q ratio (after / before)",
) -> None:
    """
    リモート 2Q ゲート数の削減率（after/before）の散布図。
    partitioned_lc みたいな「リモート重視」系の効果を見る用。
    """
    plt.figure()
    by_pipe: Dict[str, List[float]] = {}

    for s in samples:
        if s.remote_before is None or s.remote_before == 0:
            continue
        ratio = s.remote_after / s.remote_before
        by_pipe.setdefault(s.pipeline, []).append(ratio)

    for name, ratios in by_pipe.items():
        arr = _as_array(ratios)
        plt.scatter(range(len(arr)), arr, label=name)

    plt.xlabel("sample index")
    plt.ylabel("remote_after / remote_before")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_remote_ratio_vs_density(
    samples: List[BenchmarkSample],
    title: str = "Remote 2Q ratio vs CX density",
) -> None:
    """
    remote_after / remote_before を cx_density ごとに平均し、
    パイプライン別に線グラフで描く。
    """
    plt.figure()

    by_pipe_density: Dict[tuple[str, float], List[float]] = {}
    densities_set = set()
    pipelines_set = set()

    for s in samples:
        if s.remote_before is None or s.remote_before == 0:
            continue
        ratio = s.remote_after / s.remote_before
        key = (s.pipeline, float(s.cx_density))
        by_pipe_density.setdefault(key, []).append(ratio)
        densities_set.add(float(s.cx_density))
        pipelines_set.add(s.pipeline)

    densities_sorted = sorted(densities_set)

    for pipe in sorted(pipelines_set):
        xs: List[float] = []
        ys: List[float] = []
        for dens in densities_sorted:
            key = (pipe, dens)
            if key not in by_pipe_density:
                continue
            arr = _as_array(by_pipe_density[key])
            xs.append(dens)
            ys.append(arr.mean())
        if xs:
            plt.plot(xs, ys, marker="o", label=pipe)

    plt.xlabel("cx_density")
    plt.ylabel("mean(remote_after / remote_before)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()


def plot_effective_cost_ratio(
    samples: List[BenchmarkSample],
    title: str = "Effective cost ratio (after / before)",
) -> None:
    """
    eff_cost_after / eff_cost_before のヒストグラム。
    """
    plt.figure()
    by_pipe: Dict[str, List[float]] = {}

    for s in samples:
        if s.eff_cost_before is None or s.eff_cost_before == 0:
            continue
        ratio = s.eff_cost_after / s.eff_cost_before
        by_pipe.setdefault(s.pipeline, []).append(ratio)

    for name, ratios in by_pipe.items():
        arr = _as_array(ratios)
        plt.hist(arr, bins=20, alpha=0.5, label=name)

    plt.xlabel("eff_cost_after / eff_cost_before")
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
   