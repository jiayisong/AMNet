# MonoAMNet: Three-Stage Real-Time Monocular 3D Object Detection With Adaptive Methods
Code implementation of my paper [AMNet](https://ieeexplore.ieee.org/document/10843993). The code is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d).
## Environment Installation

### Create a new conda environment
```shell
conda create -n amnet python=3.7
conda activate amnet
```
### Install the [pytorch](https://pytorch.org/get-started/previous-versions/)
```shell
# CUDA 11.1
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# CUDA 10.2
pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```
### Install dependent libraries
```shell
git clone https://github.com/jiayisong/AMNet.git
cd AMNet
cd mmcv
MMCV_WITH_OPS=1 pip install -e .  # 安装好 mmcv-full
cd ..
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
pip install mmsegmentation==0.14.1
cd mmdetection3d
pip install -v -e .  # or "python setup.py develop"
```
## Dataset Download
### KITTI
Download images from the [kitti](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d), including 
*Download left color images of object data set (12 GB)*
and
*Download right color images, if you want to use stereo information (12 GB)*.

The labeled files need to be converted, and for convenience I uploaded the converted files directly. It is [kitti_label.zip](https://drive.google.com/file/d/1B0v6gn00houqtYUqlSdpK2MQEZQQhqBT/view?usp=sharing).

Unzip the image file and organize it and the label file as follows.
```
kitti
├── testing
│   ├── image_2
|   |   ├──000000.png
|   |   ├──000001.png
|   |   ├──''''
├── training
│   ├── image_2
|   |   ├──000000.png
|   |   ├──000001.png
|   |   ├──''''
├── kitti_infos_test.pkl
├── kitti_infos_train.pkl
├── kitti_infos_trainval.pkl
├── kitti_infos_val.pkl
├── kitti_infos_test_mono3d.coco.json
├── kitti_infos_train_mono3d.coco.json
├── kitti_infos_trainval_mono3d.coco.json
├── kitti_infos_val_mono3d.coco.json
```
### NuScenes
Download images from the [NuScenes](https://www.nuscenes.org/nuscenes#download).

In our experiment, we used images from the FRONT CAMERA, and we provided the corresponding labels for the forward-facing camera. It is [nuscenes_label.zip](https://drive.google.com/file/d/13tcu7hKZ9gMti-Jkb3hrTMbxrVw01753/view?usp=sharing)
## Citation

If you find this project useful in your research, please consider citing:

```latex
@ARTICLE{10843993,
  author={Pan, Huihui and Jia, Yisong and Wang, Jue and Sun, Weichao},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={MonoAMNet: Three-Stage Real-Time Monocular 3D Object Detection With Adaptive Methods}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Monocular 3D object detection;deep learning;autonomous driving;optimizer},
  doi={10.1109/TITS.2025.3525772}
}
```

