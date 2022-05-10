# Multimodal dynamic feature fusion 3D Object Detection

## Contributions

1. Supports end-to-end training

2. An effective integration strategy  

3. Low cost of semantic branching memory (no use of 2D semantic networks and labels)  

4. Without complex pre-processing, the original point cloud and image are directly learned

## Install

The Environment：

* Linux (tested on Ubuntu 18.04)
* Python 3.6+
* PyTorch 1.2+

a. Clone the repository.

```shell
git clone https://github.com/bcy1252/Adaptive3DNet.git
```

b. Install the dependent python libraries like `easydict`,`tqdm`, `tensorboardX ` etc.

c. Build and install the `pointnet2_lib`, `iou3d`, `roipool3d` libraries by executing the following command:

```shell
sh build_and_install.sh
```

## Dataset preparation

Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 

```
EPNet
├── data
│   ├── KITTI
│   │   ├── ImageSets
│   │   ├── object
│   │   │   ├──training
│   │   │      ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │   ├──testing
│   │   │      ├──calib & velodyne & image_2
├── lib
├── pointnet2_lib
├── tools
```


## Metric

The 3D Detection results of Car on Recall 40:

| Class | Easy  | Moderate | Hard  |  mAP  |
| :---: | :---: | :------: | :---: | :---: |
|  Car  | 90.86 |  79.59   | 76.93 | 82.46 |

## Implementation

For convenience, shell scripts are used, so look at the code for more information

### Training

Run net for single gpu(GPU memory >= 8GB):

```shell
cd ./tools
sh run_train_model.sh #One head, default train car class 
```

Run net for multiple gpu:

```shell
cd ./tools && vim run_train_model.sh
CUDA_VISIBLE_DEVICES=0,1,2... #modification this line
```

### Testing

```shell
cd ./tools
sh run_eval_model.sh #KITTI Benchmark
```

