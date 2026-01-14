# **Reinforcement Learning Framework for Radical Optimization**

This repository provides a complete implementation of a reinforcement learning (RL) framework for ranking and optimizing radical candidates. The pipeline is designed to train lightweight neural agents that guide radical selection under descriptor constraints.  

The repository includes both modular **Python codes** and interactive **Jupyter notebooks** for reproducibility, benchmarking, and downstream analysis.

---

## **Repository Structure**

This repo is structured as a **script-first research codebase** (Option A):
- reusable code under `src/rlcdft/`
- runnable entry points under `scripts/`  
- **notebook/** → Contains Jupyter notebooks for interactive experimentation and visualization.  

Additional folder:

- **data/** → Input CSV files (training, validation, benchmark).
- - **Radicals/** → Contains all the radical chemical structures. 

---
## Install

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

For local development (so `import rlcdft` works):

```bash
pip install -e .
```

## TD3 training (ω tuning)

Input: a CSV containing the descriptor columns (default set below), including the ω column
(default: `Electrophilicity`).

Run:

```bash
python scripts/train_td3.py \
  --data path/to/dataset.csv \
  --outdir outputs/td3_run1 \
  --target-omega 1.0 \
  --success-thr 0.05 \
  --episodes 500 \
  --steps 50
```

Outputs:
- `actor_td3.pth`, `critic_td3.pth`
- `history.json`
- `train_curves.rewards.png`, `train_curves.success.png`
- `episode_trace.png`
- `run_config.json`

### Default state columns

```text
Electronegativity, Hardness, Electrophilicity, q(N),
f-, f+, f0, s-, s+, s0, s+/s-, s-/s+, s(2)
```

You can override them with `--state-cols` and `--omega-col`.

## Motif enrichment

Compute enrichment of motif classes between a full motif table (Excel) and a top subset (CSV):

```bash
python scripts/motif_enrichment.py \
  --motif-xlsx path/to/Radical_Name_List.xlsx \
  --top-csv path/to/top20.csv \
  --out-csv outputs/motif_enrichment.csv
```

---

This will load your training and validation data from **data/**. 


## **License**

This project is released under the MIT License. See LICENSE for details.

---

## **Citation**

If you use this repository, please cite:





This repo is structured as a **script-first research codebase** (Option A):
- reusable code under `src/rlcdft/`
- runnable entry points under `scripts/`



## Notes

- This codebase intentionally avoids hard-coded paths (Colab/Drive) and runs from CLI.
- The TD3 environment included here applies a **scalar action** that edits ω only.
  If you later want a multi-action environment that edits additional descriptors,
  you can extend `ContinuousRadicalEnv` in `src/rlcdft/envs.py`.
