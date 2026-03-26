# stage_1_Reallocation of Quantum Circuit


量子回路を簡約し、特に**分散・分割実行を意識した 2 量子ビットゲート削減**を評価するためのリポジトリです。

このリポジトリでは、以下のような比較を行えます。

- Qiskit `transpile` のみを使ったベースライン
- PyZX の `full_reduce` を用いた ZX-based circuit reduction
- 回路の相互作用グラフを **k-way partition** し、remote gate を意識して簡約する手法
- `remote/local` の重み付きコスト  
  `eff_cost = w_remote * (#remote 2Q) + w_local * (#local 2Q)`  
  を最小化するローカルサーチ

---

## Features

- ランダム量子回路の生成
- 指定基底への分解 (`cx`, `rx`, `ry`, `rz` など)
- 2 量子ビットゲート数のカウント
- ZX-calculus ベースの回路簡約
- プロセッサ分割を考慮した remote/local 2Q ゲート評価
- 分割付き ZX ローカルサーチ
- ベンチマーク実行と可視化

---

### Experimental notebooks

- `trial.ipynb`  
  総合ベンチマーク用 Notebook。ランダム回路・クラスタ構造回路・CX 密度掃引に対して、各最適化パイプラインの 2Q gate 削減率、runtime、remote ratio、effective cost を可視化して比較する。

- `trial2.ipynb`  
  effective cost を中心にした統計比較用 Notebook。ランダム回路およびクラスタ構造回路に対して各パイプラインを評価し、`eff_cost_after / eff_cost_before` の統計量や手法間の勝率を集計する。

- `trial3.ipynb`  
  解析用 Notebook。Qiskit transpiler の各 optimization pass 後の 2Q gate 数・depth の変化を callback で追跡し、さらに自作の ZX-based local search with cancellation を Qiskit baseline と直接比較する。

## Repository Structure

```text
.
├── benchmark.py      # ベンチマーク実行、ランダム回路生成、可視化
├── circuit_opt.py    # 回路簡約・分割・等価性検証ロジック
├── requirements.txt  # 依存ライブラリ
├── trial.ipynb       #   探索的ベンチマーク用の Notebook です。ランダム回路や構造を持つ回路に対して複数の最適化パイプラインを適用し、2量子ビットゲート数、回路深さ、実行時間、remote gate 比率、effective cost などの指標を可視化しながら比較します。手法がどのような条件で有効に働くかを大まかに把握したい場合に使います。
├── trial2.ipynb      # 統計比較用の Notebook です。各最適化手法について、`eff_cost_after / eff_cost_before` のようなコスト比や改善量を集計し、平均・分散・勝率などの観点から比較します。可視化よりも、提案手法が baseline に対してどの程度有利かを数値的に評価したい場合に適しています。
├── trial3.ipynb      # 解析・デバッグ用の Notebook です。Qiskit transpiler の各 pass 実行後に 2量子ビットゲート数や回路深さの変化を追跡し、どの最適化 pass が結果に寄与しているかを調べます。あわせて、自作の ZX-based local search や cancellation を含む最適化手法を baseline と直接比較し、改善するケース・改善しないケースの原因分析にも使えます。
└── LICENSE
```
## Key Functions

### `circuit_opt.py`

#### `load_qasm(qasm_path_or_str)`
QASM ファイルパスまたは QASM 文字列から `QuantumCircuit` を読み込みます。  
外部の回路データをこのリポジトリの最適化フローへ渡す際の入口になります。

#### `count_two_qubit_gates(circ, twoq_names=("cx", "cz", "swap", "iswap", "ecr"))`
回路中の 2 量子ビットゲート数を数えます。  
最適化前後で回路の複雑さを比較するための基本指標として使います。

#### `unitary_equiv(qc1, qc2, atol=1e-8)`
2つの回路がグローバル位相を除いて等価かを厳密に確認します。  
小〜中規模回路の検証に向いています。

#### `fast_equiv_probe(qc1, qc2, shots=8, seed=0, atol=1e-6)`
大規模回路向けの軽量な等価性チェックです。  
厳密検証が重い場合に、近似的な確認として使います。

#### `zx_reduce_circuit(qc)`
PyZX の `full_reduce` を用いて回路を ZX-calculus ベースで簡約します。  
分割を考慮しない標準的な ZX-based reduction の基礎関数です。

#### `build_interaction_graph(qc, twoq_names=("cx", "cz", "swap", "iswap", "ecr"))`
回路中の 2 量子ビット相互作用を、重み付き無向グラフとして構築します。  
後続の回路分割で、どの量子ビット同士の結合が強いかを表現するために使います。

#### `kway_partition(G, sizes, max_passes=10)`
サイズ制約付きの k-way partition を行います。  
量子ビットを複数プロセッサに割り当てる前処理として利用します。

#### `count_remote_twoq(qc, part_of, twoq_names=("cx", "cz", "swap", "iswap", "ecr"))`
与えられた partition に対して、2 量子ビットゲートを `remote` と `local` に分類して数えます。  
分散実行時の通信コストを近似評価するための基本関数です。

#### `eff_cost(qc, part_of, w_remote=5.0, w_local=1.0, twoq_names=("cx", "cz", "swap", "iswap", "ecr"))`
`remote` / `local` 2Q ゲートに重みを付けた proxy cost を計算します。

`eff_cost = w_remote * (#remote 2Q) + w_local * (#local 2Q)`

partition-aware optimization の主評価指標です。

#### `zx_local_search_min_cost(...)`
ZX グラフ上で局所探索を行い、`eff_cost` の改善を狙う最適化関数です。  
partition-aware optimization の中核となる実装です。

#### `zx_local_search_min_cost_with_cancel(...)`
`zx_local_search_min_cost` に 2Q cancellation を組み合わせた改良版です。  
現在の partition-aware pipeline で中心的に使われる関数です。

---

### `benchmark.py`

#### `random_cx_circuit(num_qubits, depth, cx_density, seed)`
指定した量子ビット数・深さ・CX 密度に基づいてランダム回路を生成します。  
最適化手法の比較実験や tutorial 用のサンプル生成に使えます。

#### `decompose_to_basis(circuit, basis_gates=("cx", "rx", "ry", "rz"), optimization_level=1)`
回路を指定した基底ゲートセットへ分解します。  
評価対象のゲート集合をそろえたいときに便利です。

#### `transpile_only_pipeline(qc_in, ...)`
Qiskit `transpile` のみを用いるベースラインです。  
提案手法と比較する際の基準として使います。

#### `lc_pipeline(qc_in, ...)`
PyZX の `full_reduce` を用いた標準的な ZX-based optimization pipeline です。  
partition-aware ではない ZX 簡約の比較対象として利用できます。

#### `partitioned_lc_pipeline(qc_in, ...)`
分割を意識した旧版の partition-aware pipeline です。  
以前の remote-aware reduction を比較したい場合に使います。

#### `partitioned_lc_pipeline_v2(qc_in, ...)`
ZX ローカルサーチと rollback を用いた新しい partition-aware pipeline です。  
このリポジトリで最も重要な最適化フローであり、回路分割と remote cost の削減を同時に扱います。

#### `benchmark_pipeline_on_random(...)`
指定した pipeline をランダム回路群に対して一括評価します。  
パラメータ掃引や統計比較に使うベンチマーク関数です。

#### `plot_twoq_reduction_hist(samples, title=None)`
2 量子ビットゲート削減率の分布を可視化します。

#### `plot_runtime_vs_size(samples, title=None)`
回路サイズと実行時間の関係を可視化します。

#### `plot_remote_ratio(samples, title=None)`
`remote_after / remote_before` を用いて remote gate の削減傾向を可視化します。

#### `plot_effective_cost_ratio(samples, title=None)`
`eff_cost_after / eff_cost_before` の分布を可視化し、partition-aware optimization の効果を評価します。



## 謝辞
本プロジェクトは New Energy and Industrial Technology Development Organization (NEDO) JPNP23003 により支援されたものです。