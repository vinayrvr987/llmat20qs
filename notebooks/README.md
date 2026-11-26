# Notebooks

Use the **llmat20qs analysis** kernel (setup in `requirements-analysis.txt`).

| Notebook | Purpose |
|---|---|
| `v2_evaluation.ipynb` | Evaluates the final v2 runs (E1–E6, 3,000 games) against the proposal's research questions: pairing and role effects, self-play, pick randomness and scouting, question information (E5/E6), answer accuracy, and lies/noise vs theory. Numbers come from `src/analysis/evaluation.py`. |
| `v1_salvage_evaluation.ipynb` | Evaluates the archived v1 E1–E4 logs (free-retry guess bug): exact results for unaffected games, win-rate bounds for affected ones, and bug-independent measures. Numbers come from `src/analysis/v1_salvage.py`. |

The archived pilot notebook is preserved under `archive/pilot/notebooks/`.
