# ReCAST

Code for **ReCAST: Receiver-Cache-Aware Selective Refresh for Bandwidth-Efficient Cooperative Perception**, submitted to ICASSP 2027.

ReCAST aligns historical BEV features to the current ego frame and refreshes regions according to the difference between fresh features and the receiver cache. The experiments use OPV2V, V2XSet, and V2X-Seq SPD.

## Setup

Install a CUDA-enabled environment following the [OpenCOOD installation guide](https://opencood.readthedocs.io/). Place [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) at `baselines/code/OpenCOOD`; the experiment scripts use commit `31ba16025da27ffe4e336f011290dfbc66f9a1f1`.

```bash
mkdir -p baselines/code
git clone https://github.com/DerrickXuNu/OpenCOOD.git baselines/code/OpenCOOD
git -C baselines/code/OpenCOOD checkout 31ba16025da27ffe4e336f011290dfbc66f9a1f1
```

## Data

Prepare OPV2V in OpenCOOD format, then create the scene split listed in `icassp2027_revision/dataset_views/opv2v_internal/split.json`:

```bash
python3 scripts/prepare_opv2v_split.py \
  --source /path/to/OPV2V/test \
  --destination icassp2027_revision/dataset_views/opv2v_internal
```

Place OpenCOOD-format V2XSet test scenes in `icassp2027_revision/dataset_views/v2xset/test`. Convert V2X-Seq SPD to the same paired-agent format:

```bash
python3 icassp2027_revision/experiments/adapt_spd_opencood.py \
  --spd-root /path/to/V2X-Seq-SPD \
  --output icassp2027_revision/dataset_views/spd_opencood
```

## Run

Run these commands from the repository root. PointPillars is trained on the OPV2V scene split; V2XSet and SPD use that checkpoint for zero-shot evaluation. Output files are saved under `outputs/`.

<!--
### Train PointPillars

```bash
python3 icassp2027_revision/experiments/train_point_pillar.py \
  --config icassp2027_revision/experiments/point_pillar_internal.yaml \
  --output icassp2027_revision/experiments/point_pillar_run \
  --seed 2027 --workers 4
```

The selected checkpoint is `icassp2027_revision/experiments/point_pillar_run/best.pth`.
-->

### Three-dataset evaluation

The three commands below use the same checkpoint and write separate JSON files containing AP, communication fraction, and protocol metadata:

```bash
mkdir -p outputs
E=icassp2027_revision/experiments
K=$E/point_pillar_run/best.pth

python3 $E/evaluate_learned_direct.py --config $E/point_pillar_internal.yaml \
  --checkpoint $K --dataset-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --dataset-label opv2v_internal_test --modes ego,full,direct \
  --max-cav 2 --budget 0.10 --output outputs/opv2v_main.json

python3 $E/evaluate_learned_direct.py --config $E/point_pillar_v2xset_zero_shot.yaml \
  --checkpoint $K --dataset-dir icassp2027_revision/dataset_views/v2xset/test \
  --dataset-label v2xset_zero_shot --modes ego,full,direct \
  --max-cav 2 --budget 0.10 --output outputs/v2xset_main.json

python3 $E/evaluate_learned_direct.py --config $E/point_pillar_spd_zero_shot.yaml \
  --checkpoint $K --dataset-dir icassp2027_revision/dataset_views/spd_opencood \
  --dataset-label v2xseq_spd_zero_shot --modes ego,full,direct \
  --max-cav 2 --budget 0.10 --output outputs/spd_main.json
```

### Policy comparison and component ablation

The policy command evaluates Periodic, Tx-Delta, Random-B, Oldest-first, and ReCAST with the same detector. The ablation command calibrates the four fixed-threshold settings on validation scenes, then evaluates all eight component combinations on held-out scenes.

```bash
E=icassp2027_revision/experiments
K=$E/point_pillar_run/best.pth
mkdir -p outputs

python3 $E/evaluate_learned_direct.py --config $E/point_pillar_internal.yaml \
  --checkpoint $K --dataset-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --dataset-label opv2v_internal_test \
  --modes periodic,tx_delta,random_b,oldest_first,direct \
  --max-cav 2 --budget 0.10 --seed 2027 \
  --output outputs/opv2v_refresh_policies.json

python3 scripts/run_factorial_ablation.py --config $E/point_pillar_internal.yaml \
  --checkpoint $K \
  --validation-dir icassp2027_revision/dataset_views/opv2v_internal/validate \
  --test-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --budget 0.10 --output-dir outputs/factorial
```

### Budget and cache-age curves

```bash
E=icassp2027_revision/experiments
K=$E/point_pillar_run/best.pth
mkdir -p outputs

python3 $E/evaluate_budget_curve.py --config $E/point_pillar_internal.yaml \
  --checkpoint $K --dataset-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --budgets 0.02,0.05,0.10,0.25,0.50 --methods periodic,direct \
  --include-baselines --output outputs/opv2v_budget_curve.json

python3 $E/evaluate_exact_age.py --config $E/point_pillar_internal.yaml \
  --checkpoint $K --dataset-dir icassp2027_revision/dataset_views/opv2v_internal/test \
  --gaps 0,1,2,4,6,8 --output outputs/opv2v_exact_age.json

python3 $E/make_learned_figures_side_by_side.py \
  --budget-json outputs/opv2v_budget_curve.json \
  --age-json outputs/opv2v_exact_age.json --output-dir outputs/figures
```

Use `evaluate_budget_curve.py` with `point_pillar_v2xset_zero_shot.yaml` or `point_pillar_spd_zero_shot.yaml` and the corresponding dataset directory to sweep budgets on the other two datasets.

<!--
## Dependencies

Datasets, checkpoints, and third-party baseline code are obtained from their original sources. See [THIRD_PARTY.md](THIRD_PARTY.md).
-->
