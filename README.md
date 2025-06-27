# [Less is More: Efficient Image Vectorization with Adaptive Parameterization](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Less_is_More_Efficient_Image_Vectorization_with_Adaptive_Parameterization_CVPR_2025_paper.html)

Official implementation of **AdaVec**, from the following paper:

Less is More: Efficient Image Vectorization with Adaptive Parameterization, CVPR 2025

[[Project Website](https://zhaokaibo830.github.io/adavec/)]

![![title]](imgs/pipeline.png?raw=true)

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
