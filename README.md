# ReCAST

Code for **ReCAST: Receiver-Cache-Aware Selective Refresh for Bandwidth-Efficient Cooperative Perception**, submitted to ICASSP 2027.

ReCAST aligns a collaborator's cached bird's-eye-view features with the current ego frame, scores the difference between fresh features and the receiver cache, and refreshes selected regions under a communication budget. The repository includes PointPillars training, ReCAST evaluation, controlled refresh policies, cache age analysis, and plotting scripts.

## Installation

Use a Linux system with a CUDA-capable GPU and an environment compatible with [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). Follow the [OpenCOOD installation guide](https://opencood.readthedocs.io/) to install PyTorch, CUDA extensions, Open3D, NumPy, PyYAML, and the remaining OpenCOOD requirements. The scripts were developed with OpenCOOD commit `31ba16025da27ffe4e336f011290dfbc66f9a1f1`.

From the ReCAST repository root, place OpenCOOD at the path expected by the experiment scripts:

```bash
mkdir -p baselines/code
git clone https://github.com/DerrickXuNu/OpenCOOD.git baselines/code/OpenCOOD
git -C baselines/code/OpenCOOD checkout 31ba16025da27ffe4e336f011290dfbc66f9a1f1
```

OpenCOOD is an external dependency; its source code, datasets, and detector checkpoints are not redistributed here. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Data

Experiments use OPV2V, V2XSet, and V2X-Seq SPD. Train PointPillars on the documented scene-disjoint OPV2V split; use that checkpoint for zero-shot evaluation on V2XSet and SPD. Prepare the datasets in OpenCOOD format before running evaluation.

### OPV2V

Obtain an OpenCOOD-format OPV2V extraction. The following helper creates symlinks for the internal train, validation, and held-out test scene lists in `icassp2027_revision/dataset_views/opv2v_internal/split.json`:

```bash
python3 scripts/prepare_opv2v_split.py \
  --source /path/to/OPV2V/test \
  --destination icassp2027_revision/dataset_views/opv2v_internal
```

This is an internal feasibility split, not the official OPV2V benchmark split. The helper does not copy point clouds or overwrite existing paths.

### V2XSet

Place the OpenCOOD-format V2XSet test scenes here:

```text
icassp2027_revision/dataset_views/v2xset/test/<scene>/<agent>/<frame>.yaml
icassp2027_revision/dataset_views/v2xset/test/<scene>/<agent>/<frame>.pcd
```

The evaluator reads this directory through `--dataset-dir`; no V2XSet training is required for the zero-shot protocol.

### V2X-Seq SPD

Obtain the official SPD sequence, including vehicle and infrastructure LiDAR, calibration, cooperative labels, and `cooperative/data_info.json`. Convert it to a paired OpenCOOD-format sequence:

```bash
python3 icassp2027_revision/experiments/adapt_spd_opencood.py \
  --spd-root /path/to/V2X-Seq-SPD \
  --output icassp2027_revision/dataset_views/spd_opencood
```

The adapter creates YAML metadata and symlinks to the source point clouds. The example experiment uses this adapted sequence for zero-shot evaluation.

## Run

Run commands below from the ReCAST repository root. They write JSON results to `outputs/`, which is excluded from Git. `--budget 0.10` means a nominal 10% regional refresh budget; the realized transmitted-feature fraction, including initialization, is recorded in each result. `--max-frames 0` means all frames in the selected view.

### 1. Train the detector on OPV2V

The cache experiments use one PointPillars checkpoint. Train it once on the internal OPV2V scene split:

```bash
python3 icassp2027_revision/experiments/train_point_pillar.py \
  --config icassp2027_revision/experiments/point_pillar_internal.yaml \
  --output icassp2027_revision/experiments/point_pillar_run \
  --seed 2027 --workers 4
```

The training script writes `best.pth`, `latest.pth`, `config.yaml`, `metadata.json`, and `metrics.jsonl` to `point_pillar_run/`. To use an existing compatible checkpoint, pass its path with `--checkpoint` in the evaluation commands below.

### 2. Evaluate ReCAST on all three datasets

Use `--modes ego,full,direct` to evaluate the single-agent reference, full feature communication, and ReCAST on the same dataset view. Give each dataset a distinct `--output` path.

OPV2V held-out scenes:

```bash
mkdir -p outputs
python3 icassp2027_revision/experiments/evaluate_learned_direct.py \
  --config icassp2027_revision/experiments/point_pillar_internal.yaml \
  --checkpoint icassp2027_revision/experiments/point_pillar_run/best.pth \
  --dataset-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --dataset-label opv2v_internal_test \
  --modes ego,full,direct --max-cav 2 --budget 0.10 \
  --output outputs/opv2v_main.json
```

V2XSet zero-shot:

```bash
python3 icassp2027_revision/experiments/evaluate_learned_direct.py \
  --config icassp2027_revision/experiments/point_pillar_v2xset_zero_shot.yaml \
  --checkpoint icassp2027_revision/experiments/point_pillar_run/best.pth \
  --dataset-dir icassp2027_revision/dataset_views/v2xset/test \
  --dataset-label v2xset_zero_shot \
  --modes ego,full,direct --max-cav 2 --budget 0.10 \
  --output outputs/v2xset_main.json
```

V2X-Seq SPD zero-shot:

```bash
python3 icassp2027_revision/experiments/evaluate_learned_direct.py \
  --config icassp2027_revision/experiments/point_pillar_spd_zero_shot.yaml \
  --checkpoint icassp2027_revision/experiments/point_pillar_run/best.pth \
  --dataset-dir icassp2027_revision/dataset_views/spd_opencood \
  --dataset-label v2xseq_spd_zero_shot \
  --modes ego,full,direct --max-cav 2 --budget 0.10 \
  --output outputs/spd_main.json
```

Each result contains average precision at IoU thresholds 0.3, 0.5, and 0.7, the requested budget, the realized feature-value fraction, and dataset protocol metadata. V2XSet and SPD use the OPV2V-trained checkpoint without target-dataset training.

### 3. Compare refresh policies and components

For the controlled policy comparison on held-out OPV2V, run all methods with one detector, split, and nominal budget:

```bash
python3 icassp2027_revision/experiments/evaluate_learned_direct.py \
  --config icassp2027_revision/experiments/point_pillar_internal.yaml \
  --checkpoint icassp2027_revision/experiments/point_pillar_run/best.pth \
  --dataset-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --dataset-label opv2v_internal_test \
  --modes periodic,tx_delta,random_b,oldest_first,direct \
  --max-cav 2 --budget 0.10 --seed 2027 \
  --output outputs/opv2v_refresh_policies.json
```

For the eight combinations of temporal transport, receiver-cache residual, and Top-B selection, calibrate fixed thresholds on validation scenes and evaluate on held-out scenes:

```bash
python3 scripts/run_factorial_ablation.py \
  --config icassp2027_revision/experiments/point_pillar_internal.yaml \
  --checkpoint icassp2027_revision/experiments/point_pillar_run/best.pth \
  --validation-dir icassp2027_revision/dataset_views/opv2v_internal/validate \
  --test-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --budget 0.10 --output-dir outputs/factorial
```

The runner saves validation calibrations for the four configurations without Top-B and a JSON result for each of the eight test configurations. Codes `A/B/C` correspond to transport, receiver-cache residual, and Top-B respectively; `factor_111` enables all three.

### 4. Sweep budgets and measure exact cache age

The budget sweep evaluates Direct and periodic refresh and also computes ego-only and full-communication references:

```bash
python3 icassp2027_revision/experiments/evaluate_budget_curve.py \
  --config icassp2027_revision/experiments/point_pillar_internal.yaml \
  --checkpoint icassp2027_revision/experiments/point_pillar_run/best.pth \
  --dataset-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --dataset-label opv2v_internal_test \
  --budgets 0.02,0.05,0.10,0.25,0.50 \
  --methods periodic,direct --include-baselines \
  --output icassp2027_revision/experiments/learned_budget_curve.json
```

For V2XSet and SPD, use the same evaluator and checkpoint with their dataset-specific configuration and a separate output path:

```bash
python3 icassp2027_revision/experiments/evaluate_budget_curve.py \
  --config icassp2027_revision/experiments/point_pillar_v2xset_zero_shot.yaml \
  --checkpoint icassp2027_revision/experiments/point_pillar_run/best.pth \
  --dataset-dir icassp2027_revision/dataset_views/v2xset/test \
  --dataset-label v2xset_zero_shot \
  --budgets 0.10,0.20,0.30,0.40,0.50 \
  --methods periodic,direct --include-baselines \
  --output outputs/v2xset_budget_curve.json

python3 icassp2027_revision/experiments/evaluate_budget_curve.py \
  --config icassp2027_revision/experiments/point_pillar_spd_zero_shot.yaml \
  --checkpoint icassp2027_revision/experiments/point_pillar_run/best.pth \
  --dataset-dir icassp2027_revision/dataset_views/spd_opencood \
  --dataset-label v2xseq_spd_zero_shot \
  --budgets 0.10,0.20,0.30,0.40,0.50 \
  --methods periodic,direct --include-baselines \
  --output outputs/spd_budget_curve.json
```

The exact-age study evaluates ages `0,1,2,4,6,8` on a common OPV2V frame set:

```bash
python3 icassp2027_revision/experiments/evaluate_exact_age.py \
  --config icassp2027_revision/experiments/point_pillar_internal.yaml \
  --checkpoint icassp2027_revision/experiments/point_pillar_run/best.pth \
  --dataset-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --dataset-label opv2v_internal_test \
  --gaps 0,1,2,4,6,8 \
  --output icassp2027_revision/experiments/learned_exact_age.json
```

Generate the two-panel figure from those JSON files:

```bash
python3 icassp2027_revision/experiments/make_learned_figures_side_by_side.py \
  --budget-json icassp2027_revision/experiments/learned_budget_curve.json \
  --age-json icassp2027_revision/experiments/learned_exact_age.json \
  --output-dir outputs/figures
```

The plotting command writes PDF and PNG files. These commands evaluate ReCAST with the shared detector. The public-method rows in the paper use separate official implementations and released checkpoints, so the scripts here do not regenerate those rows.

## Data and third-party software

Datasets, pretrained weights, and third-party baseline implementations are not included. Obtain them from their official distribution channels and follow their terms. See [THIRD_PARTY.md](THIRD_PARTY.md).
