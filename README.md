# Less is More: Efficient Image Vectorization with Adaptive Parameterization（CVPR 2025）

[[Project Website](https://zhaokaibo830.github.io/adavec/)]

We propose AdaVec, an efficient image vectorization method with adaptive parametrization, where the paths and control points can be adjusted dynamically based on the complexity of the input raster image.

![![title]](imgs/pipeline.png?raw=true)

This work is largely inspired by [LIVE](https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization),[O&R](https://github.com/ajevnisek/optimize-and-reduce) and [SGLIVE](https://github.com/Rhacoal/SGLIVE)

## Installation
We suggest users to use the conda for creating new python environment. 

```bash
git clone https://github.com/IMU-Group/AdaVec.git
cd AdaVec
conda create -n adavec python=3.8
conda activate adavec
pip install -r requirements.txt

cd DiffVG
git submodule update --init --recursive
python setup.py install
cd ..
```

## Run Experiments 
```bash
conda activate adavec
cd AdaVec
python main.py 
```

## Reference

    @inproceedings{zhao2025less,
	  title     = {Less is More: Efficient Image Vectorization with Adaptive Parameterization},
	  author    = {Zhao, Kaibo and Bao, Liang and Li, Yufei and Su, Xu and Zhang, Ke and Qiao, Xiaotian},
	  booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference},
 	  pages     = {18166--18175},
	  year      = {2025}
	}

## Acknowledgement
Our implementation is mainly based on the [LIVE](https://github.com/Picsart-AI-Research/LIVE-Layerwise-Image-Vectorization),[O&R](https://github.com/ajevnisek/optimize-and-reduce) and [SGLIVE](https://github.com/Rhacoal/SGLIVE) codebase. We gratefully thank the authors for their wonderful works.
This work was also partially supported by Guangdong Basic and Applied Basic Research Foundation (No. 2022A1515110740), National Natural Science Foundation of China (No. 62302356, No. 62172316), Key Research and Development Program of Hebei Province, China (No. 23310302D), and Key Research and Development Program of Shaanxi Province, China (No. 2024GX-ZDCYL-01-11).
