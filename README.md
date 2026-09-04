# Representation Learning for Sample-Efficient CATE Estimation by Leveraging Multiple Outcomes

This codebase reproduces the experiments in *Representation Learning for Sample-Efficient CATE Estimation by Leveraging Multiple Outcomes*.

## Installation

The reported experiments used Python 3.9. To replicate the environment used to run the code, follow the commands below:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

`pip install -e .` installs this package and the dependencies declared in `pyproject.toml`. Installing `requirements.txt` first uses the exact dependency versions used for the paper.

## Paper-to-code map

| Paper artifact | Result | Experiment |
|---|---|---|
| Figure `fig:main_results` | Main normalized-PEHE and policy-value results on the synthetic and semi-synthetic datasets | Main synthetic run, raw-$X$ comparison, and semi-synthetic $\dim(\phi)=10$ run |
| Figures `fig:synth_appendix_xl` and `fig:synth_appendix_dml`; Tables `tab:synth_pehe_norm_xl`, `tab:synth_policy_norm_20_xl`, `tab:synth_pehe_norm_dml`, `tab:synth_policy_norm_20_dml`, and `tab:synthetic_raw_x_comparison` | Synthetic results by CATE learner and outcome target | `paper_main.yaml` and `raw_x_comparison.yaml` |
| Figures `fig:alpha-sweep-dml-pehe` and `fig:alpha-sweep-dml-policy` | Sufficiency-(ii) violation sweep | `alpha_sweep_dml_varmatched.yaml` |
| Figures `fig:delta-sweep-dml-pehe` and `fig:delta-sweep-dml-policy` | Surrogacy-violation sweep | `delta_sweep_dml_varmatched.yaml` |
| Figures `fig:m-sweep-dml-pehe` and `fig:m-sweep-dml-policy` | Representation-dimension sweep on the synthetic dataset | `m_sweep.yaml` |
| Figure `fig:semisynth_phi10_full`; Tables `tab:corrected_semisynth_phi10_pehe` and `tab:corrected_semisynth_phi10_policy` | Complete semi-synthetic comparison at $\dim(\phi)=10$ | Predicted-outcome run with `--phi-dim 10` |
| Figure `fig:semisynth_phi5_full`; Table `tab:corrected_semisynth_phi_sensitivity` | Semi-synthetic representation-dimension sensitivity | Predicted-outcome run with `--phi-dim 5` |
| Figure `fig:semisynth_phi10_true_y`; Tables `tab:corrected_semisynth_phi10_pehe_true_y` and `tab:corrected_semisynth_phi10_policy_true_y` | Semi-synthetic sensitivity using observed experimental $Y$ | Observed-outcome runs with `--phi-dim 5` and `--phi-dim 10` |
| Table `tab:semisynth-diagnostics` | Cohort-overlap and representation-sufficiency diagnostics | `diagnose_corrected_semisynth_assumptions.py` |

The commands below first reproduce the experiment data and then regenerate the corresponding paper artifacts.

## Run the experiments

### Main-body results

#### Synthetic experiment

Run the main synthetic experiment and the comparison between the two raw-covariate outcome models:

```bash
python scripts/run_slice.py configs/slices/paper_main.yaml \
  --device cuda --no-progress

python scripts/run_slice.py configs/slices/raw_x_comparison.yaml \
  --device cuda --no-progress
```

These commands reproduce the synthetic panels in Figure `fig:main_results`, Figures `fig:synth_appendix_xl` and `fig:synth_appendix_dml`, and the synthetic tables listed in the paper-to-code map. The first command evaluates both CATE learners and both outcome targets. Each configuration uses one fixed DGP and ten paired data draws.

#### Semi-synthetic experiment

The main semi-synthetic experiment uses a ten-dimensional representation, predicted outcomes from the historical outcome model, and three-fold DML:

```bash
python scripts/run_semisynthetic.py \
  --cache-dir data/semisynth_cache \
  --phi-dim 10 \
  --seeds 10 \
  --cate-learner econml_dml \
  --outcome-source predicted \
  --ate-calibration none \
  --output results/semisynthetic_phi10.parquet
```

This command reproduces the semi-synthetic panels in Figure `fig:main_results`, Figure `fig:semisynth_phi10_full`, and Tables `tab:corrected_semisynth_phi10_pehe` and `tab:corrected_semisynth_phi10_policy`.

The semi-synthetic source data and processed cache are not included in this release while we complete the remaining preprocessing and redistribution checks. Aggregate results used to generate the paper's figures and tables are included under `results/paper/` and contain no row-level observations or predictions.

### Appendix results

The sufficiency-violation, surrogacy-violation, and representation-dimension experiments use:

```bash
python scripts/run_slice.py configs/slices/alpha_sweep_dml_varmatched.yaml \
  --device cuda --no-progress

python scripts/run_slice.py configs/slices/delta_sweep_dml_varmatched.yaml \
  --device cuda --no-progress

python scripts/run_slice.py configs/slices/m_sweep.yaml \
  --device cuda --no-progress
```

These commands reproduce Figures `fig:alpha-sweep-dml-pehe`, `fig:alpha-sweep-dml-policy`, `fig:delta-sweep-dml-pehe`, `fig:delta-sweep-dml-policy`, `fig:m-sweep-dml-pehe`, and `fig:m-sweep-dml-policy`. The corrected violation sweeps keep the scale of treatment-effect heterogeneity fixed as the violation parameter changes. The X-learner and observed-outcome sensitivity analyses for the main synthetic setting are produced by `paper_main.yaml` above.

For the semi-synthetic representation-dimension and observed-outcome sensitivities, run:

```bash
python scripts/run_semisynthetic.py \
  --cache-dir data/semisynth_cache \
  --phi-dim 5 --seeds 10 \
  --cate-learner econml_dml \
  --outcome-source predicted --ate-calibration none \
  --output results/semisynthetic_phi5.parquet

python scripts/run_semisynthetic.py \
  --cache-dir data/semisynth_cache \
  --phi-dim 10 --seeds 10 \
  --cate-learner econml_dml \
  --outcome-source true --ate-calibration none \
  --output results/semisynthetic_true_y_phi10.parquet

python scripts/run_semisynthetic.py \
  --cache-dir data/semisynth_cache \
  --phi-dim 5 --seeds 10 \
  --cate-learner econml_dml \
  --outcome-source true --ate-calibration none \
  --output results/semisynthetic_true_y_phi5.parquet
```

Run the cohort-overlap and representation-sufficiency diagnostics with:

```bash
python scripts/diagnose_corrected_semisynth_assumptions.py \
  --cache-dir data/semisynth_cache \
  --phi-dim 10
```

These commands reproduce Figures `fig:semisynth_phi5_full` and `fig:semisynth_phi10_true_y` and Tables `tab:corrected_semisynth_phi_sensitivity`, `tab:corrected_semisynth_phi10_pehe_true_y`, `tab:corrected_semisynth_phi10_policy_true_y`, and `tab:semisynth-diagnostics`.

## Regenerate the figures and tables

Regenerate Figure `fig:main_results` from the aggregate results included in `results/paper/`:

```bash
python scripts/plot_harmonized_main_figure.py \
  --synthetic-results results/paper/synthetic_main.csv \
  --synthetic-raw-comparison results/paper/synthetic_raw_x_comparison.csv \
  --semisynthetic-results results/paper/semisynthetic_phi10.csv \
  --output results/paper/main_figure.pdf
```

Regenerate Figures `fig:synth_appendix_xl`, `fig:synth_appendix_dml`, `fig:alpha-sweep-dml-pehe`, `fig:alpha-sweep-dml-policy`, `fig:delta-sweep-dml-pehe`, `fig:delta-sweep-dml-policy`, `fig:m-sweep-dml-pehe`, and `fig:m-sweep-dml-policy`, together with Tables `tab:synth_pehe_norm_xl`, `tab:synth_policy_norm_20_xl`, `tab:synth_pehe_norm_dml`, `tab:synth_policy_norm_20_dml`, and `tab:synthetic_raw_x_comparison`:

```bash
python scripts/export_synthetic_paper_tables.py \
  --input results/paper/synthetic_main.csv \
  --raw-comparison results/paper/synthetic_raw_x_comparison.csv \
  --output-dir results/paper/synthetic/main

python scripts/export_synthetic_paper_panels.py \
  --main results/paper/synthetic_main.csv \
  --alpha results/paper/synthetic_alpha_dml_varmatched.csv \
  --delta results/paper/synthetic_delta_dml_varmatched.csv \
  --m-sweep results/paper/synthetic_m_sweep.csv \
  --m-sweep-max-failures 4 \
  --output-root results/paper/synthetic
```

Regenerate Figures `fig:semisynth_phi10_full`, `fig:semisynth_phi5_full`, and `fig:semisynth_phi10_true_y`, together with Tables `tab:corrected_semisynth_phi10_pehe`, `tab:corrected_semisynth_phi10_policy`, `tab:corrected_semisynth_phi_sensitivity`, `tab:corrected_semisynth_phi10_pehe_true_y`, and `tab:corrected_semisynth_phi10_policy_true_y`:

```bash
python scripts/export_semisynth_paper_tables.py \
  --phi5 results/paper/semisynthetic_phi5.csv \
  --phi10 results/paper/semisynthetic_phi10.csv \
  --output-dir results/paper/semisynthetic/tables

python scripts/export_semisynth_paper_tables.py \
  --phi5 results/paper/semisynthetic_true_y_phi5.csv \
  --phi10 results/paper/semisynthetic_true_y_phi10.csv \
  --output-dir results/paper/semisynthetic/tables_true_y \
  --label-suffix _true_y \
  --caption-note ' using observed experimental $Y$'

python scripts/plot_corrected_semisynth_fresh.py \
  --input results/paper/semisynthetic_phi10.csv \
  --output-dir results/paper/semisynthetic/phi10

python scripts/plot_corrected_semisynth_fresh.py \
  --input results/paper/semisynthetic_phi5.csv \
  --output-dir results/paper/semisynthetic/phi5

python scripts/plot_corrected_semisynth_fresh.py \
  --input results/paper/semisynthetic_true_y_phi10.csv \
  --output-dir results/paper/semisynthetic/true_y_phi10
```

## Compute

The reported experiments were run with SLURM. Synthetic runs use a GPU pretraining job followed by CPU evaluation arrays. The semi-synthetic submission script requests one GPU, four CPU cores, and 8 GB of memory for each seed. The repository includes the corresponding submission and collection scripts under `scripts/cluster/`. We do not report an end-to-end runtime as the experiments were split across arrays and the logs do not provide a reliable wall-clock estimate.

