# YOLO v2
## Description

This is a personal implementation of YOLO v2(You Only Look Once) written as a Python package with a CLI tool, using Keras (TensorFlow backend). YOLO is a convolutional real-time object detection system. More info about it can be found on the original papers [[1]](#1) [[2]](#2).

I should also mention [this](https://https://www.maskaravivek.com/post/yolov2/) blog from which I took lots of inspiration. 

## Requirements
Requirements for this project can be found on requirements.txt and installed via:

```
pip3 install requirements.txt
```

## Usage
An example of usage of the package embedded in Python code can be found on the main.py file.

A user must define a configuration file `.yaml` (set by default to `config.yaml` in the execution directory). See [config.yaml](config.yaml) for a more detailed description of what this file should contain.

For running the CLI tool, see:

```
python3 -m yolo --help
```
## References
<a id="1">[1]</a> 
J. Redmon, S. Divvala, R.Girshick and A. Farhadi. You only look once: Unified, real-time object detection. [*arXiv preprint arXiv:1506.02640*](https://arxiv.org/abs/1506.02640), 2015

<a id="2">[2]</a> 
J. Redmon and A. Farhadi. YOLO9000: Better, faster, stronger. [*arXiv preprint arXiv:1612.08242*](https://arxiv.org/abs/1612.08242), 2016