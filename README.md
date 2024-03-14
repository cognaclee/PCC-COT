# PCC-COT

**Point Cloud Compression via Constrained Optimal Transport**

(Official Pytorch Implementation, the code is modified from [D-PCC](https://github.com/yunhe20/D-PCC))

## Introduction
Point cloud compression (PCC) algorithms are typically designed to achieve the lowest possible distortion at a given low bit rate. However, the perceptual quality is often neglected. To tackle this, we innovatively regard PCC as a constrained optimal transport (COT) problem and proposea novel data-driven method to take the balance of distortion, perception, and the bit rate. Specifically, our method adopts a discriminator to measure the perceptual loss, and a generator to measure the optimal mapping from the original point cloud distribution to the reconstructed distribution.
## Results
* Quantitative results

![image](https://github.com/cognaclee/PCC-COT/blob/main/Docs/imgs/Quantitative_results.jpg)

* Qualitative results

![image](https://github.com/cognaclee/PCC-COT/blob/main/Docs/imgs/Qualitative_results.jpg)

## Installation

* Install the following packages

```
python==3.7
torch==1.7.1
torchvision==0.8.2
CUDA==11.0
numpy==1.20.3
open3d==0.9.0.0
einops==0.3.2
scikit-learn==1.0.1
compressai
ninja
pickle
argparse
tensorboard
```

## Data Preparation

First download the [ShapeNetCore](https://shapenet.org/download/shapenetcore) v1 and [SemanticKITTI](http://semantic-kitti.org/dataset.html#download) datasets, and then divide them into non-overlapping blocks.

* ShapeNet

```
# install the `Manifold' program
cd ./dataset
git clone https://github.com/hjwdzh/Manifold
cd Manifold && mkdir build
cd build 
cmake .. -DCMAKE_BUILD_TYPE=Release
make 
cd ..

# divide into blocks
python prepare_shapenet.py --date_root path/to/shapenet
```

* SemanticKITTI

```
python prepare_semantickitti.py --data_root path/to/semantickitti
```

## Train

```
# shapenet
python train.py --dataset shapenet
# semantickitti
python train.py --dataset semantickitti
```

## Test
```
# shapenet
python test.py --dataset shapenet --model_path path/to/model
# semantickitti
python test.py --dataset semantickitti --model_path path/to/model
```

The decompressed patches and full point clouds will also be saved at `./output/experiment_id/pcd` by default.

## Acknowledgments

Our code is built upon the following repositories: [D-PCC](https://github.com/yunhe20/D-PCC), [DEPOCO](https://github.com/PRBonn/deep-point-map-compression), [PAConv](https://github.com/CVMI-Lab/PAConv), [Point Transformer](https://github.com/qq456cvb/Point-Transformers) and [MCCNN](https://github.com/viscom-ulm/MCCNN), thanks for their great work.

## Citation

If you find our project is useful, please consider citing:
