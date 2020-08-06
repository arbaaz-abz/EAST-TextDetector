# EAST: An Efficient and Accurate Scene Text Detector

### Introduction
This is a tensorflow re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2).
The features are summarized blow:
+ Only **RBOX** part is implemented.
+ A fast Locality-Aware NMS in C++ provided by the paper's author.
+ Differences from original paper
	+ Use ResNet-50 rather than PVANET
	+ Use dice loss (optimize IoU of segmentation) rather than balanced cross entropy
	+ Use linear learning rate decay rather than staged learning rate decay

Thanks to the author ([@argman](https://github.com/argman))
Original [paper](https://arxiv.org/abs/1704.03155v2) if you find this useful.

### Download
1. Models trained on ICDAR 2013 (training set) + ICDAR 2015 (training set): [GoogleDrive](https://drive.google.com/open?id=0B3APw5BZJ67ETHNPaU9xUkVoV0U)
2. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### Test
```
python eval.py --test_data_path=demo_images/ --gpu_list=0 --checkpoint_path=model/east_icdar2015_resnet_v1_50_rbox/ \
--output_dir=output/
```

a text file will be then written to the output path.
