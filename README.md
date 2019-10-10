# DeepGRU: Deep Gesture Recognition Utility

Official PyTorch implementation of [DeepGRU](https://arxiv.org/abs/1810.12514).

<p align="center">
  <img width="500" src="https://github.com/Maghoumi/DeepGRU/blob/master/images/DeepGRU.png"/>
</p>

## Getting Started

I've tried my best to make the code work out-of-the-box, and provide an extensible framework to ease experimentation.

### Prerequisites

The list of requirements for this project is extremely short:
- Python v3.5+
- [PyTorch v1.2+](https://pytorch.org/)
- [Numpy](https://numpy.org/) (will be installed along PyTorch)
- (Optional) [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) with an NVIDIA GPU (for faster training, although CPU-only training still works)

All the requirements are included in `requirements.txt`.

### Running the Code:

Running the code involves 3 easy steps:

1) Obtain the sources
```
git clone https://github.com/Maghoumi/DeepGRU.git
cd DeepGRU
```

2) Install the dependencies (make sure the correct `pip` is used)
```
pip install -r requirements.txt
```

3) Run the code (make sure the correct `python` (v3.5+) is used)
```
python main.py
```

The above code will download the [SBU Kinect Interaction](https://www3.cs.stonybrook.edu/~kyun/research/kinect_interaction/index.html) dataset and run the standard 5-fold cross-validation experiments. The dataset will be downloaded to `DeepGRU/data` and the run results will be dumped under `DeepGRU/logs`.

The training progress will be showed in the standard output, and you should see an average recognition accuracy of about 95.5% if all folds run to completion.


## A Few Notes

### Omission of GPSR Implementation

At the time of writing, [GPSR](https://dl.acm.org/citation.cfm?id=2984525) is [patented in the United States](https://patents.google.com/patent/US20180018533A1/en) and its inclusion in any project would impose a [restrictive license](https://github.com/ISUE/Jackknife/blob/master/LICENSE) on any open source code. As such, I've decided to omit GPSR from this repository to allow unrestricted usage of DeepGRU. You can implement GPSR yourself by referring to the pseudocode in the appendix of the original paper.

### Result Reproduction

We used PyTorch v0.4 in our original experiments, and we encountered slightly different results with more recent versions of PyTorch (even with the same random seeds). The results also vary depending on what GPU is used for training. As mentioned above, our implementation of GPSR is omitted from this repository due to licensing issues. Considering all these, the exact results of our paper may not be easily reproducible.

## Citing

If you find this code or our paper useful, kindly please cite our work:

```
@article{maghoumi2018deepgru,
  title={DeepGRU: Deep Gesture Recognition Utility},
  author={Maghoumi, Mehran and LaViola Jr, Joseph J},
  journal={arXiv preprint arXiv:1810.12514},
  year={2018}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

