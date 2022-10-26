# PCSC-COT

**Sample and Compress Point Cloud via Constrained Optimal Transport**

(Official Pytorch Implemention, the code is modified from [D-PCC](https://github.com/yunhe20/D-PCC))

## Introduction

## Results
* Quantitative results

* Qualitative results

## Installation

* Install the following packages



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
