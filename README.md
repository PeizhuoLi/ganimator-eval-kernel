# GANimator evaluation kernel

## Usage

This repository contains the C++ implementation of the dynamic programming kernel of evaluation part of GANimator.

Prerequisites:

- Eigen

To install the kernel, first clone this repository recursively:


```bash
git clone --recursive https://github.com/PeizhuoLi/ganimator-eval-kernel.git
```

Then, use pip to install this module in your virtual environment for ganimator:

```bash
conda activate ganimator
pip install ./ganimator-eval-kernel
```

## Acknowledgements

This repository is based on [cmake_example for pybind11](https://github.com/pybind/cmake_example).
