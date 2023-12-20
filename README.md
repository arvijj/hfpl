# High-fidelity Pseudo-labels

This repository is the official implementation of the paper [__High-fidelity Pseudo-labels for Boosting Weakly-Supervised Segmentation__, WACV, 2024](https://arxiv.org/pdf/2304.02621.pdf) \[1\]. It contains the implementation of the binomial-based importance sampling loss (ISL) and feature similarity loss (FSL) for SEAM \[2\]. The losses are implemented in `tool/probutils.py` and used in `train_cam_<voc/coco>.py`.

## Installation

Install with conda:
- [Install miniconda](https://docs.conda.io/en/latest/miniconda.html)
- `conda create -n hfpl python=3.6`
- `conda activate hfpl`
- `pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html`

Install on a singularity container:
- [Install singularity](https://sylabs.io/guides/3.0/user-guide/quick_start.html)
- `singularity build --fakeroot pytorch.sif pytorch.def`

Download and configure data:
- VOC: `./config_voc.sh`
- COCO: `./config_coco.sh` (requires `make`)

Download the ImageNet pretrained weights `ilsvrc-cls_rna-a1_cls1000_ep-0001.params` from [here](https://drive.google.com/drive/folders/1y3WjoQLgqbR4q7u9KWPjj4enlLvbn_-P?usp=sharing) and put them in a new folder named `pretrained`.

## Training and evaluation

Run training/inference/evaluation on all three stages CAM/AffinityNet/final (note: writes to `./exp/` and overwrites previous runs):
- VOC 2012: `./run_voc.sh`
- COCO 2014: `./run_coco.sh`

Note that the ImageNet pretrained model path needs to be set manually in `lib/net/backbone/resnet38d.py`, which is `./pretrained` by default.

For training on multiple GPUs, update the `GPUS` field in the config file `voc12/config_voc2012.py` or `coco/config_coco2014.py` to match the number of available GPUs on your system.

## Acknowledgements

This code was based on the following repositories:
- [YudeWang/SEAM](https://github.com/YudeWang/SEAM) \[2\]
- [YudeWang/semantic-segmentation-codebase](https://github.com/YudeWang/semantic-segmentation-codebase) \[2\]
- [jiwoon-ahn/irn](https://github.com/jiwoon-ahn/irn) \[3\]
- [davisvideochallenge/davis2017-evaluation](https://github.com/davisvideochallenge/davis2017-evaluation/) \[4\]
- [cocodataset/cocoapi](https://github.com/cocodataset/cocoapi)

## References

\[1\] Arvi Jonnarth, Yushan Zhang, and Michael Felsberg. High-fidelity Pseudo-labels for Boosting Weakly-Supervised Segmentation. WACV, 2024.

\[2\] Yude Wang, Jie Zhang, Meina Kan, Shiguang Shan, and Xilin Chen. Self-Supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation. CVPR, 2020.

\[3\] Jiwoon Ahn and Suha Kwak. Learning Pixel-Level Semantic Affinity with Image-Level Supervision for Weakly Supervised Semantic Segmentation. CVPR, 2018.

\[4\] Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation. CVPR, 2016.
