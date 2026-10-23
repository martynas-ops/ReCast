# Third-party software

ReCAST's experiment scripts use the following software as external dependencies:

- **OpenCOOD**: <https://github.com/DerrickXuNu/OpenCOOD>, tested with commit `31ba16025da27ffe4e336f011290dfbc66f9a1f1`. The local OpenCOOD snapshot carries an academic software license that prohibits redistribution and further transfer. It is therefore not vendored here. Clone it separately and comply with its license.
- **Where2comm**: <https://github.com/MediaBrain-SJTU/where2comm>, commit `d3432523973b8c11df71e8e85750b2cb6fbe9dbb`.
- **How2comm**: <https://github.com/ydk122024/How2comm>, commit `8d357abdcb265ed4db5e4c6d9ee5400ea5799abe`.
- **CoSDH**: <https://github.com/Xu2729/CoSDH>, commit `b270b69fe133c0c1ca2044ad072cd77fc21863fb`.
- **CPCoCo**: <https://github.com/saki-mobile/cpcoco>, commit `2f5e6f9e487f78d4bc6ca43da2fcfd1b24ec09ff`.
- **PyTorch, NumPy, Open3D, PyYAML, Matplotlib**: used by the detector training, evaluation, and plotting scripts. Install versions compatible with the selected OpenCOOD environment and CUDA toolkit.

The exact source revisions are machine-readable in `baselines/manifest.json`.
Small patch files add dataset-path overrides, fixed-frame limits, and payload
accounting to the public inference programs. The CPCoCo audit patch also adds
the explicit energy-ranked spatial budget used for its five operating points.
No released weight is modified or retrained. Obtain third-party checkpoints and
datasets from their official sources and follow their terms.
