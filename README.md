# ReCAST

Official experiment release for **ReCAST: Receiver-Cache-Aware Selective Refresh for Bandwidth-Efficient Cooperative Perception**, submitted to ICASSP 2027.

ReCAST aligns historical bird's-eye-view features to the current ego frame, maintains region-wise age and validity, and refreshes only expired or high-residual regions under a communication budget. The release covers OPV2V, V2XSet, and V2X-Seq SPD; the detector is frozen during cache-policy evaluation.

## Setup

Create a CUDA environment following the [OpenCOOD installation guide](https://opencood.readthedocs.io/), then fetch the exact OpenCOOD revision:

```bash
python3 scripts/setup_public_baselines.py --method OpenCOOD
python3 scripts/setup_public_baselines.py --method OpenCOOD --verify-only
```

The released ReCAST checkpoint is `checkpoints/recast_pointpillar_opv2v.pth`. Its SHA-256 digest is recorded in `checkpoints/SHA256SUMS`.

## Data

Prepare OPV2V in OpenCOOD format and create the published scene split:

```bash
python3 scripts/prepare_opv2v_split.py \
  --source /path/to/OPV2V/test \
  --destination icassp2027_revision/dataset_views/opv2v_internal
```

Place V2XSet test scenes in OpenCOOD format. Convert V2X-Seq SPD with:

```bash
python3 icassp2027_revision/experiments/adapt_spd_opencood.py \
  --spd-root /path/to/V2X-Seq-SPD \
  --output icassp2027_revision/dataset_views/spd_opencood
```

## Run

The following command runs the three-dataset accuracy and budget sweeps, matched-budget policy comparison, exact-age curve, component ablation, and paper figures. Set `--max-frames` to a small positive value for a smoke test; zero evaluates every frame.

```bash
python3 scripts/reproduce_recast.py \
  --checkpoint checkpoints/recast_pointpillar_opv2v.pth \
  --opv2v icassp2027_revision/dataset_views/opv2v_internal/test \
  --opv2v-validation icassp2027_revision/dataset_views/opv2v_internal/validate \
  --v2xset icassp2027_revision/dataset_views/v2xset/test \
  --spd icassp2027_revision/dataset_views/spd_opencood \
  --output-dir outputs/reproduction --device cuda
```

Motion-only lifetime prediction and geometric survival rate are evaluated from synchronized object tracks and the exact-age output:

```bash
python3 icassp2027_revision/experiments/evaluate_motion_prediction.py \
  --dataset-root icassp2027_revision/dataset_views/opv2v_internal/test \
  --exact-age-json outputs/reproduction/opv2v_exact_age.json \
  --output outputs/reproduction/opv2v_motion_prediction.json
```

For public baselines, clone the pinned repositories and apply only the included inference/audit instrumentation:

```bash
python3 scripts/setup_public_baselines.py \
  --method Where2comm --method How2comm --method CoSDH --method CPCoCo \
  --apply-audit-patches
python3 scripts/run_public_audit.py --help
```

The fixed-scene audit source JSON files are included. Regenerate the communication table with:

```bash
python3 icassp2027_revision/experiments/build_cumulative_communication_table.py
python3 icassp2027_revision/experiments/build_cpcoco_table2.py
```

Run release checks before evaluation:

```bash
python3 scripts/verify_release.py
```

<!--
### Train PointPillars

python3 icassp2027_revision/experiments/train_point_pillar.py \
  --config icassp2027_revision/experiments/point_pillar_internal.yaml \
  --output icassp2027_revision/experiments/point_pillar_run \
  --seed 2027 --workers 4

### Dependencies

Datasets, third-party checkpoints, and public baseline implementations are obtained from their official sources. See THIRD_PARTY.md.
-->
