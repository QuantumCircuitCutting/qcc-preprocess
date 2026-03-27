from pathlib import Path

import networkx as nx
import numpy as np
import pyzx as zx
from qiskit import QuantumCircuit, transpile
from qiskit.qasm2 import dumps
from qiskit.quantum_info import Operator


# ============================================================
# QASM / Circuit 基本ユーティリティ
# ============================================================

def load_qasm(qasm_path_or_str: str) -> QuantumCircuit:
    """ファイルパス or QASM文字列を受け取り、QuantumCircuitを返す。"""
    s = qasm_path_or_str

    # 明らかにQASMソースなら即文字列扱い
    if "OPENQASM" in s or "\n" in s:
        return QuantumCircuit.from_qasm_str(s)

    # それ以外はパスとして扱ってみる
    try:
        p = Path(s)
        if p.exists():
            return QuantumCircuit.from_qasm_file(str(p))
    except OSError:
        # ファイル名として扱えない → QASM文字列
        return QuantumCircuit.from_qasm_str(s)

    # exists() が False だった時も文字列扱い
    return QuantumCircuit.from_qasm_str(s)


def count_two_qubit_gates(
    circ: QuantumCircuit,
    twoq_names=("cx", "cz", "iswap", "ecr", "rxx", "ryy", "rzz", "swap"),
) -> int:
    """指定された2量子ビットゲートの個数を数える。"""
    count = 0
    for instr in circ.data:  # CircuitInstruction
        name = instr.operation.name.lower()
        if name in twoq_names:
            count += 1
    return count


def cz_to_cx(qc: QuantumCircuit) -> QuantumCircuit:
    """
    CZをCX + H(target) に変換する。
    Qiskit 1.x の CircuitInstruction API に対応。
    """
    new = QuantumCircuit(qc.num_qubits, qc.num_clbits)
    for instr in qc.data:
        op = instr.operation
        qubs = instr.qubits
        clbs = instr.clbits
        name = op.name.lower()
        if name == "cz":
            ctrl, targ = qubs
            new.h(targ)
            new.cx(ctrl, targ)
            new.h(targ)
        else:
            new.append(op, qubs, clbs)
    return new


def simplify_single_qubit(
    qc: QuantumCircuit,
    basis=("u", "cx"),
) -> QuantumCircuit:
    """
    単一量子ビットゲートをtranspileで吸収・整理する。
    basis_gates はターゲットデバイスに合わせて調整。
    """
    return transpile(qc, basis_gates=list(basis), optimization_level=3)


# ============================================================
# 等価性チェック
# ============================================================

def unitary_equiv(
    qc1: QuantumCircuit,
    qc2: QuantumCircuit,
    atol: float = 1e-8,
) -> tuple[bool, float]:
    """
    グローバル位相同値性の厳密チェック（小〜中規模用）。
    戻り値: (等価かどうか, Frobenius距離)

    数値エラーに対して少し安全側になるようガードを入れている。
    """
    U1 = Operator(qc1).data
    U2 = Operator(qc2).data

    # そもそも次元が違えばアウト
    if U1.shape != U2.shape:
        return False, float("inf")

    with np.errstate(all="ignore"):
        M = np.conjugate(U2).T @ U1
        tr = np.trace(M)

        # trace が 0 近傍だと angle が不安定なのでガード
        if np.isclose(tr, 0.0):
            phase = 0.0
        else:
            phase = np.angle(tr)

        diff = U1 - np.exp(1j * phase) * U2
        dist = np.linalg.norm(diff, ord="fro")

    if not np.isfinite(dist):
        # 数値的に壊れたら非等価扱い＋NaN距離
        return False, float("nan")

    return (dist < atol, float(dist))


def fast_equiv_probe(
    qc1: QuantumCircuit,
    qc2: QuantumCircuit,
    shots: int = 64,
    seed: int = 1,
    atol: float = 1e-8,
) -> bool:
    """
    Monte Carlo 型の等価性テスト（大規模回路向け）。
    計算基底のランダムベクトルに対する作用が一致するかをチェック。
    """
    rng = np.random.default_rng(seed)
    n = qc1.num_qubits
    dim = 1 << n
    U1 = Operator(qc1).data
    U2 = Operator(qc2).data

    # 位相合わせのために |0...0> で基準を取る
    v0 = np.zeros((dim,), dtype=complex)
    v0[0] = 1.0
    w1 = U1 @ v0
    w2 = U2 @ v0
    phase = np.angle(np.vdot(w2, w1))

    for _ in range(shots):
        idx = rng.integers(0, dim)
        v = np.zeros((dim,), dtype=complex)
        v[idx] = 1.0
        r1 = U1 @ v
        r2 = np.exp(1j * phase) * (U2 @ v)
        if np.linalg.norm(r1 - r2) >= atol:
            return False
    return True


# ============================================================
# 通常の ZX-based 簡約
# ============================================================

def zx_reduce_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Qiskit circuit -> QASM2 -> ZX graph -> full_reduce -> 抽出 -> Qiskit.
    (Qiskit >=1.x + PyZX >=1.x 想定)
    """
    # Qiskit → QASM → PyZX Circuit
    qasm_str = dumps(qc)
    zx_circ = zx.Circuit.from_qasm(qasm_str)

    # ZX graph化 & full_reduce
    g = zx_circ.to_graph()
    zx.full_reduce(g)

    # 回路抽出
    extracted = zx.extract.extract_circuit(g)

    # PyZX → Qiskit Circuit（QASM経由）
    reduced_qasm = extracted.to_qasm()
    return QuantumCircuit.from_qasm_str(reduced_qasm)


# ============================================================
# プロセッサ分割 / リモートゲート評価 用のユーティリティ
# ============================================================

def build_interaction_graph(
    qc: QuantumCircuit,
    twoq_names=("cx", "cz", "iswap", "ecr", "rxx", "ryy", "rzz", "swap"),
) -> nx.Graph:
    """
    2量子ビットゲートの結合を重み付き無向グラフとして構築する。
    ノード: 物理キュビット index
    辺重み: そのペアに現れる 2q ゲートの回数
    """
    G = nx.Graph()
    for i in range(qc.num_qubits):
        G.add_node(i)

    for inst in qc.data:
        name = inst.operation.name.lower()
        if name in twoq_names:
            a = qc.find_bit(inst.qubits[0]).index
            b = qc.find_bit(inst.qubits[1]).index
            w = 1 + (G[a][b]["weight"] if G.has_edge(a, b) else 0)
            G.add_edge(a, b, weight=w)
    return G


def count_remote_twoq(
    qc: QuantumCircuit,
    part_of: dict[int, int],
    twoq_names=("cx", "cz", "iswap", "ecr", "rxx", "ryy", "rzz", "swap"),
) -> tuple[int, int]:
    """
    分割(part_of)に対して、ローカル/リモートの2量子ビットゲート数を数える。
    """
    remote = 0
    local = 0
    for inst in qc.data:
        name = inst.operation.name.lower()
        if name in twoq_names:
            a = qc.find_bit(inst.qubits[0]).index
            b = qc.find_bit(inst.qubits[1]).index
            if part_of[a] != part_of[b]:
                remote += 1
            else:
                local += 1
    return remote, local


# ---------------- k-way partition (サイズ制約付き簡易FM) ----------------

def _greedy_seed_partition(G: nx.Graph, sizes: list[int]) -> dict[int, int]:
    """
    初期割当て:
    ノードを重み付き次数の高い順に見て、
    その時点で内部結合重みが最大になるパーティションへ置く。
    """
    n = G.number_of_nodes()
    k = len(sizes)
    assert sum(sizes) == n

    part_of: dict[int, int] = {}
    fill = [0] * k

    nodes = sorted(G.nodes(), key=lambda v: G.degree(v, weight="weight"), reverse=True)
    for v in nodes:
        scores = []
        for p in range(k):
            if fill[p] >= sizes[p]:
                scores.append((-1e9, p))
                continue
            score = sum(G[v][u]["weight"] for u in G.neighbors(v) if part_of.get(u) == p)
            scores.append((score, p))
        _, best = max(scores)
        part_of[v] = best
        fill[best] += 1

    return part_of


def _delta_cut_if_move(
    G: nx.Graph,
    v: int,
    from_p: int,
    to_p: int,
    part_of: dict[int, int],
) -> float:
    """
    v を from_p → to_p に動かしたときのカット重み変化量 Δcut を返す。
    負なら改善。
    """
    delta = 0.0
    for u in G.neighbors(v):
        w = G[v][u]["weight"]
        pu = part_of[u]
        before_cut = (pu != from_p)
        after_cut = (pu != to_p)
        if after_cut and not before_cut:
            delta += w
        elif not after_cut and before_cut:
            delta -= w
    return delta


def kway_partition(
    G: nx.Graph,
    sizes: list[int],
    max_passes: int = 10,
) -> dict[int, int]:
    """
    サイズ制約付き k-way 分割。
    - sizes: 各パーティションの目標サイズ（合計が |V| に等しいこと）
    - 簡易FM反復でカット重みを削減。
    """
    assert sum(sizes) == G.number_of_nodes()
    k = len(sizes)

    part_of = _greedy_seed_partition(G, sizes)
    fill = [0] * k
    for v, p in part_of.items():
        fill[p] += 1

    for _ in range(max_passes):
        improved = False
        nodes = sorted(G.nodes(), key=lambda v: G.degree(v, weight="weight"), reverse=True)

        for v in nodes:
            p0 = part_of[v]
            best_p = p0
            best_gain = 0.0

            for p in range(k):
                if p == p0 or fill[p] >= sizes[p]:
                    continue
                dcut = _delta_cut_if_move(G, v, p0, p, part_of)
                if dcut < best_gain:  # よりカットが減る方向へ
                    best_gain, best_p = dcut, p

            if best_p != p0:
                part_of[v] = best_p
                fill[p0] -= 1
                fill[best_p] += 1
                improved = True

        if not improved:
            break

    return part_of


# ---------------- ZXグラフ用のラベル伝播 & LCコスト近似 ----------------

def label_spiders_by_partition(
    g: zx.Graph,
    part_of_qubits: dict[int, int],
) -> dict[int, int]:
    """
    ZXグラフのスパイダーに対して、入力ノードのパーティション情報を
    BFS で伝播させてラベルを付与する。
    """
    from collections import deque

    part: dict[int, int] = {v: None for v in g.vertices()}

    # 入力の所属パーティションを起点に伝播
    inputs = list(g.inputs())
    for q, v in enumerate(inputs):
        part[v] = part_of_qubits[q]

    dq = deque(inputs)
    while dq:
        u = dq.popleft()
        for w in g.neighbors(u):
            if part[w] is None:
                part[w] = part[u]
                dq.append(w)

    # 孤立などでラベルが付かなかったものはとりあえず0
    for v in g.vertices():
        if part[v] is None:
            part[v] = 0

    return part


def _deg(g: zx.Graph, v: int) -> int:
    """GraphS/PyZX の差異を吸収した degree(v) 互換。"""
    try:
        return g.degree(v)
    except Exception:
        return len(list(g.neighbors(v)))


def _is_connected(g: zx.Graph, a: int, b: int) -> bool:
    """GraphS/PyZX の差異を吸収した connected(a,b) 互換。"""
    try:
        return g.connected(a, b)
    except Exception:
        nb = g.neighbors(a)
        try:
            return b in nb
        except TypeError:
            return b in list(nb)


def delta_cost_lc(
    g: zx.Graph,
    v: int,
    part: dict[int, int],
    w_remote: float = 1.0,
) -> float:
    """
    LC(v) による異パーティション間結合の変化 ΔJ を近似評価。
    負なら「リモート結合」が減る方向とみなす。
    """
    N = list(g.neighbors(v))
    if len(N) < 2:
        return 0.0

    diff_pairs_total = 0
    cut_before = 0

    for i in range(len(N)):
        pi = part[N[i]]
        for j in range(i + 1, len(N)):
            pj = part[N[j]]
            if pi != pj:
                diff_pairs_total += 1
                if _is_connected(g, N[i], N[j]):
                    cut_before += 1

    cut_after = diff_pairs_total - cut_before
    return w_remote * (cut_after - cut_before)


def cut_cost(g: zx.Graph, part: dict[int, int]) -> int:
    """
    現在の ZX グラフに対する「異パーティション間エッジ数」をざっくり測る。
    """
    J = 0
    for v in g.vertices():
        pv = part[v]
        for w in g.neighbors(v):
            if w <= v:
                continue  # 無向グラフの二重カウント防止
            if part[w] != pv:
                J += 1
    return J


def remote_aware_zx_reduce(
    qc: QuantumCircuit,
    part_of: dict[int, int],
    w_remote: float = 2.0,
    max_passes: int = 3,
    max_moves: int = 200,
) -> QuantumCircuit:
    """
    リモート結合を意識した ZX 簡約（旧版）。
    - ΔJ < 0 となる頂点に対してのみ local_comp を適用。
    - GraphS 環境でも動くように degree / connected をラップ。
    """
    g = zx.Circuit.from_qasm(dumps(qc)).to_graph()

    for _ in range(max_passes):
        part = label_spiders_by_partition(g, part_of)
        moves = 0
        improved = False

        # 次数の高いスパイダーから優先的に試す
        cand = sorted(g.vertices(), key=lambda x: _deg(g, x), reverse=True)

        for v in cand:
            if moves >= max_moves:
                break

            dJ = delta_cost_lc(g, v, part, w_remote=w_remote)
            if dJ < 0:
                zx.local_comp(g, v)
                moves += 1
                improved = True

        # 軽い簡約（環境によっては存在しないので try/except）
        try:
            zx.simplify.spider_simplify(g)
        except Exception:
            pass

        try:
            zx.simplify.pivoting(g)
        except Exception:
            pass

        if not improved:
            break

    circ = zx.extract.extract_circuit(g)
    return QuantumCircuit.from_qasm_str(circ.to_qasm())


# ============================================================
# eff_cost と ZX ローカルサーチ（新）
# ============================================================

def eff_cost(
    qc: QuantumCircuit,
    part_of: dict[int, int],
    w_remote: float = 5.0,
    w_local: float = 1.0,
    twoq_names=("cx", "cz", "iswap", "ecr", "rxx", "ryy", "rzz", "swap"),
) -> tuple[float, int, int]:
    """
    分散実行コストの proxy:
        cost = w_remote * (#remote 2Q) + w_local * (#local 2Q)

    戻り値: (cost, remote_count, local_count)
    """
    remote, local = count_remote_twoq(qc, part_of, twoq_names=twoq_names)
    cost = w_remote * remote + w_local * local
    return float(cost), remote, local


def zx_local_search_min_cost(
    qc_in: QuantumCircuit,
    sizes: list[int],
    w_remote: float = 5.0,
    w_local: float = 1.0,
    max_outer_iters: int = 3,
    max_moves_per_pass: int = 50,
    min_improve_ratio: float = 1e-3,
) -> tuple[QuantumCircuit, dict[int, int], dict[str, float]]:
    """
    ZX グラフ上で LC を局所的に適用しつつ、
    「eff_cost = w_remote*remote + w_local*local」が下がる move だけを採用するローカルサーチ。

    - qc_in:    開始回路（すでに transpile 等で整えたものを想定）
    - sizes:    k-way partition の各パーティションサイズ

    戻り値:
        (best_qc, best_part_of, stats)
    """
    # 初期回路の interaction graph & partition
    G0 = build_interaction_graph(qc_in)
    part_of = kway_partition(G0, sizes)
    best_qc = qc_in
    best_cost, best_remote, best_local = eff_cost(best_qc, part_of, w_remote, w_local)

    stats: dict[str, float] = {
        "eff_cost_initial": float(best_cost),
        "remote_initial": float(best_remote),
        "local_initial": float(best_local),
        "num_accepts": 0.0,
    }

    # ZX グラフへ変換
    g = zx.Circuit.from_qasm(dumps(best_qc)).to_graph()

    for _ in range(max_outer_iters):
        improved_this_pass = False
        moves = 0

        # 次数の高いスパイダーから順に試す
        cand_vertices = sorted(g.vertices(), key=lambda v: _deg(g, v), reverse=True)

        for v in cand_vertices:
            if moves >= max_moves_per_pass:
                break

            # まずはラフなフィルタで「明らかに悪化しそう」な頂点をスキップ
            part_label = label_spiders_by_partition(g, part_of)
            dJ = delta_cost_lc(g, v, part_label, w_remote=1.0)
            if dJ >= 0:
                continue

            # LC を試す：グラフコピーして local_comp
            g_trial = g.copy()
            zx.local_comp(g_trial, v)

            # 軽い簡約（失敗しても無視）
            try:
                zx.simplify.spider_simplify(g_trial)
            except Exception:
                pass
            try:
                zx.simplify.pivoting(g_trial)
            except Exception:
                pass

            # 回路へ戻す
            circ_trial = zx.extract.extract_circuit(g_trial)
            qc_trial = QuantumCircuit.from_qasm_str(circ_trial.to_qasm())

            # candidate ごとに partition & eff_cost を取り直して評価
            G_trial = build_interaction_graph(qc_trial)
            part_of_trial = kway_partition(G_trial, sizes)
            cost_trial, remote_trial, local_trial = eff_cost(
                qc_trial,
                part_of_trial,
                w_remote,
                w_local,
            )

            # 改善していれば採用（best を更新）
            if cost_trial < best_cost * (1.0 - min_improve_ratio):
                best_qc = qc_trial
                best_cost = cost_trial
                best_remote = remote_trial
                best_local = local_trial
                part_of = part_of_trial
                g = g_trial  # グラフも更新
                improved_this_pass = True
                stats["num_accepts"] += 1.0
                moves += 1
            # 悪化 or 誤差レベル → 採用しない（rollback）

        if not improved_this_pass:
            break

    stats.update(
        {
            "eff_cost_final": float(best_cost),
            "remote_final": float(best_remote),
            "local_final": float(best_local),
        }
    )
    return best_qc, part_of, stats


from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import InverseCancellation, CommutativeCancellation


def cancel_twoq_only(
    qc: QuantumCircuit,
    twoq_names=("cx", "cz", "iswap", "ecr", "rxx", "ryy", "rzz", "swap"),
) -> QuantumCircuit:
    """
    2量子ビットゲートの cancellation だけを狙って軽くかける。
    ZX変形の途中で挟む用途。

    注意:
      - InverseCancellation は「同じ2Qゲートが逆順で並ぶ」等を消す
      - CommutativeCancellation は可換性を使って移動して消す
    """
    # Qiskit pass は list[str] を受けるので tuple→list に
    names = [g.lower() for g in twoq_names]

    pm = PassManager(
        [
            InverseCancellation(names),
            CommutativeCancellation(names),
        ]
    )
    return pm.run(qc)


def zx_local_search_min_cost_with_cancel(
    qc_in: QuantumCircuit,
    sizes: list[int],
    w_remote: float = 5.0,
    w_local: float = 1.0,
    max_outer_iters: int = 3,
    max_moves_per_pass: int = 50,
    min_improve_ratio: float = 1e-3,
    cancel_each_trial: bool = True,
    cancel_on_accept: bool = False,
    twoq_names=("cx", "cz", "iswap", "ecr", "rxx", "ryy", "rzz", "swap"),
) -> tuple[QuantumCircuit, dict[int, int], dict[str, float]]:
    """
    zx_local_search_min_cost の改良版：
      - trial回路を eff_cost 評価する直前に 2Q cancellation を挟む。
        → Qiskit(opt3)が強い “cancellation” を探索ループに取り込む。

    パラメータ:
      - cancel_each_trial:
          True なら各 trial 評価前に cancellation を実行（推奨）
      - cancel_on_accept:
          True なら受理した best_qc にも cancellation をもう一度実行
          (通常は不要。探索の挙動が変わるのが嫌なら False のままでOK)

    戻り値:
      (best_qc, best_part_of, stats)
    """
    # 初期回路の interaction graph & partition
    G0 = build_interaction_graph(qc_in, twoq_names=twoq_names)
    part_of = kway_partition(G0, sizes)
    best_qc = qc_in
    best_cost, best_remote, best_local = eff_cost(
        best_qc, part_of, w_remote, w_local, twoq_names=twoq_names
    )

    stats: dict[str, float] = {
        "eff_cost_initial": float(best_cost),
        "remote_initial": float(best_remote),
        "local_initial": float(best_local),
        "num_accepts": 0.0,
        "num_trials": 0.0,
        "num_cancel_calls": 0.0,
    }

    # ZX グラフへ変換
    g = zx.Circuit.from_qasm(dumps(best_qc)).to_graph()

    for _ in range(max_outer_iters):
        improved_this_pass = False
        moves = 0

        # 次数の高いスパイダーから順に試す
        cand_vertices = sorted(g.vertices(), key=lambda v: _deg(g, v), reverse=True)

        for v in cand_vertices:
            if moves >= max_moves_per_pass:
                break

            # ラフなフィルタ
            part_label = label_spiders_by_partition(g, part_of)
            dJ = delta_cost_lc(g, v, part_label, w_remote=1.0)
            if dJ >= 0:
                continue

            # LC を試す
            g_trial = g.copy()
            zx.local_comp(g_trial, v)

            # 軽い簡約（失敗しても無視）
            try:
                zx.simplify.spider_simplify(g_trial)
            except Exception:
                pass
            try:
                zx.simplify.pivoting(g_trial)
            except Exception:
                pass

            # 回路へ戻す
            circ_trial = zx.extract.extract_circuit(g_trial)
            qc_trial = QuantumCircuit.from_qasm_str(circ_trial.to_qasm())

            # ★ここが追加：trial評価前に cancellation を挟む
            if cancel_each_trial:
                qc_trial = cancel_twoq_only(qc_trial, twoq_names=twoq_names)
                stats["num_cancel_calls"] += 1.0

            stats["num_trials"] += 1.0

            # candidate ごとに partition & eff_cost を取り直して評価
            G_trial = build_interaction_graph(qc_trial, twoq_names=twoq_names)
            part_of_trial = kway_partition(G_trial, sizes)
            cost_trial, remote_trial, local_trial = eff_cost(
                qc_trial, part_of_trial, w_remote, w_local, twoq_names=twoq_names
            )

            # 改善していれば採用
            if cost_trial < best_cost * (1.0 - min_improve_ratio):
                best_qc = qc_trial
                if cancel_on_accept:
                    best_qc = cancel_twoq_only(best_qc, twoq_names=twoq_names)
                    stats["num_cancel_calls"] += 1.0

                best_cost = cost_trial
                best_remote = remote_trial
                best_local = local_trial
                part_of = part_of_trial
                g = g_trial  # グラフも更新

                improved_this_pass = True
                stats["num_accepts"] += 1.0
                moves += 1
            # 採用しない場合は rollback（何もしない）

        if not improved_this_pass:
            break

    stats.update(
        {
            "eff_cost_final": float(best_cost),
            "remote_final": float(best_remote),
            "local_final": float(best_local),
        }
    )
    return best_qc, part_of, stats