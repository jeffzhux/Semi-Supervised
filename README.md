# The Dual-Expert in PyTorch

Code for the paper: "A New Semi-Supervised Learning Method for Long-Tailed Recognition with The Dual-Expert Deep Neural Networks" by Chu-Hsiang Lee, Jiunn-Lin Wu.

## Install
This project uses [virtualenv](https://docs.python.org/3/library/venv.html) to create virtual environments.
```sh
python -m venv .venv
.venv/Scripts/activate
pip install -r requirement.txt
```

# Dataset
Download or generate the datasets as follows:
- CIFAR 10 and CIFAR100: Follow the [step](https://github.com/google-research/fixmatch/blob/master/README.md#install-datasets) to download and generate balanced CIFAR10 and CIFAR100 datasets. Put it under .data/cifar10 and ./data/cifar100.

# Running experiment on Long-tailed CIFAR10, CIFAR100
Run [FixMatch](https://github.com/jeffzhux/Semi-Supervised/blob/master/fixmatch.py) ([paper](https://arxiv.org/abs/2001.07685)):
* Specify the config path via `--config`.
* Specify task via `--task`. It can be `FixMatch`, `Our` or `SL`.
* Specify mode via `--mode`. It can be `train`, `test` or `export`. If you select `test` or `export` mode, you must add `--weight`.
```sh
python -m main \
  --config ./configs/.. \
  --task=fixmatch \
  --mode=train
```

# Testing experiment on Long-tailed CIFAR10, CIFAR100
* Specify the model weight path via `--weight`.
```sh
python -m main \
  --config ./path/to/config/.. \
  --task=fixmatch \
  --mode=test \
  --weight=./path/to/model/..
```

# Results
The code reproduces the main result of the paper. These results are from [Crest](https://arxiv.org/abs/2102.09559) except for Ours.

**Results on Long-tailed CIFAR10 with 30% labeled data**
|          | gamma=50 | gamma=100 | gamma=200 |
| -------- | -------- | --------- | --------- |
| FixMatch | 81.9     | 73.1      | 64.7      |
| [CReST]  | 84.2     | 77.6      | 67.7      |
| [CReST+] | 84.9     | 79.2      | 70.5      |
| Ours     | 86.2     | 81.4      | 76.6      |

# Export model
Export the model weights to ONNX format and copy the resulting file to the designated folder. To build the web, look at [Plant_Disease_Classification
](https://github.com/jeffzhux/Plant_Disease_Classification). Please note that the image preprocess during the model forward.
```sh
python -m main \
  --config ./path/to/config/.. \
  --task=fixmatch \
  --mode=export \
  --weight=./path/to/model/..
```

# Citing this work
```bibtex
@article{chu-hsiang2023DualExpert,
    title={A New Semi-Supervised Learning Method for Long-Tailed Recognition with The Dual-Expert Deep Neural Networks},
    author={Chu-Hsiang Lee, Jiunn-Lin Wu},
    year={2023}
}
```
