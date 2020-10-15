# Unet Person Silhouette Segmantation.
## Brief description
The project includes the minimum required to run the model in inference mode.  
**Plugin Unet** *(plugin_unet)* contains the model code and auxiliary functions.  
**Stripped down library** *(sd_lib)* contains all the necessary additional functionality + methods for *further expansion*.  
**Utils** *(utils)* contains conversion scripts of models.  
________________________________________________________________________________________________
# Requirements
## Python
Install anaconda and create *conda environment* "env-name"  
```python
conda create -n env-name python=3.6
```
Install the necessary libraries sequentially in this environment:  
```python
conda install pytorch=1.5.1 torchvision -c pytorch
conda install matplotlib
conda install -c conda-forge shapely

pip install bidict
pip install opencv-python==3.4.10.35
pip install simplejson
pip install python-json-logger
pip install scikit-image
pip install prettytable
pip install jsonschema
pip install onnx
pip install onnxruntime
```
Just in case, I'll add a list of stable library version configurations:  
```python
pytorch=1.5.1
torchvision=0.6.1
matplotlib=3.3.0
shapely=1.7.0
bidict=0.19.0
opencv-python=3.4.10.35
simplejson=3.17.2
python-json-logger=0.1.8
scikit-image=0.17.2
ptable=0.9.2
jsonschema=2.6.0
onnx=1.7.0
onnxruntime=1.5.1
```
**A small note**:
Now this python project only has inference mode. The training code and all necessary extensions will be added in the future when I have more time (it requires refactoring at the moment).  Below I will add an example of what this Neural Network is capable of.
## Using
## Prediction
```python
python simple_inference.py
```
Depending on the configuration of your local machine, the model runs on the GPU or CPU, and you can see this information in the logs: 
```python
{"message": "Starting base single image inference applier init.", "timestamp": "2020-09-25T16:29:25.393Z", "level": "info"}
{"message": "Base single image inference applier init done.", "timestamp": "2020-09-25T16:29:25.393Z", "level": "info"}
{"message": "Will construct model.", "timestamp": "2020-09-25T16:29:25.414Z", "level": "info"}
{"message": "Model has been constructed (w/out weights).", "timestamp": "2020-09-25T16:29:27.330Z", "level": "info"}
{"message": "Model has been loaded into GPU(s).", "remapped_device_ids": [0], "timestamp": "2020-09-25T16:29:28.944Z", "level": "info"}
```
![](https://github.com/serjik85kg/UnetV2-pytorch-segmentation/blob/main/python/examples/47_outputs.jpg)  
In this case, the model runs on the GPU, but if it is not found, the model will run on the CPU.  
## Additional scripts  
Additional scripts lay in *utils/convertions*.    
Currently, they serve to convert the "native" *pytorch unet model* to other formats for execution from different frameworks.   
**trace_model_unet.py** - converts in torchscript format (special zip archive with .pt extension). It lets running the model in other projects, where it is possible to include Torchscript module (@TO DO: add some links about it).    
**compare_pytorch_torchscript.py** - compares "native" pytorch model and torchscript model after conversion.  
**to_onnx.py** - converts to ONNX format.  
**compare_pytorch_onnx.py** - compares "native" pytorch model and onnx model after conversion.  
**onnx_inference.py** - runs ONNX model.  

# TO DO
 - Add python code for data transformation
 - Add python code for training modes
 - Add link for pretrained model weights
 - Add building scripts
