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
├── trial.ipynb       # 実験用ノートブック
├── trial2.ipynb      # 実験用ノートブック
├── trial3.ipynb      # 実験用ノートブック
└── LICENSE
```

## 謝辞
本プロジェクトは New Energy and Industrial Technology Development Organization (NEDO) JPNP23003 により支援されたものです。