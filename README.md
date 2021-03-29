# Cars Detection

It is a simple algorithm which detect all cars present in a set of images.
The code is developed using [Detectron2](https://github.com/facebookresearch/detectron2/blob/master/README.md)

## Installation
Create a virtual environment
```
python -m verv <virtual_environment_name>
source <virtual_environment_name>/bin/activate
```
After, install all dependencies contain in file _requirements.txt_
```
pip install requirements.txt
```

## Run
First, the weight file contains. [[Link](https://drive.google.com/file/d/18BkeeoxY0GftdDffaz7StGR56n-rkpcQ/view?usp=sharing)]
Activate the virtual environment and run:
```
python test.py
```
**Parameters:**
<ul>
<li>source: The folder which containts the set of the images</li>
<li>show: If True the algorithm is going to show the images result</li>
</ul>

The algorithm is going to output a ```results.csv``` file with has two columns the first identifies the name of the image and the second the coordinate of 
bounding boxes in format ```[height, width, px_center, py_center]```