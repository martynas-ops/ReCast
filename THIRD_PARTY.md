# Third-party software

ReCAST's experiment scripts use the following software as external dependencies:

- **OpenCOOD**: <https://github.com/DerrickXuNu/OpenCOOD>, tested with commit `31ba16025da27ffe4e336f011290dfbc66f9a1f1`. The local OpenCOOD snapshot carries an academic software license that prohibits redistribution and further transfer. It is therefore not vendored here. Clone it separately and comply with its license.
- **PyTorch, NumPy, Open3D, PyYAML, Matplotlib**: used by the detector training, evaluation, and plotting scripts. Install versions compatible with the selected OpenCOOD environment and CUDA toolkit.

Datasets and pretrained checkpoints are separate assets and are not included in this repository. Obtain them from their official sources and follow their terms.
