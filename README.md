# Deep Soft Procrustes for Markerless Volumetric Sensor Alignment
Easy to use Depth Sensor Extrinsics Calibration

[Paper Page](https://vcl3d.github.io/StructureNet/)
~~[Paper]~~
~~[Supplementary Material]~~

# Requirements
- Python 3.6.7
- [PyTorch 1.2 + cuda 9.2](https://pytorch.org/get-started/previous-versions/#v120)

# Installation
(New python enviroment is highly recommended)
- Install required packages with `pip install -r requirements.txt`
**Only for training**
- Install [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) by cloning/downloading this repository, navigate to python folder and run `python setup.py install`
- Install our custom patch for disabling multisampling in pyrender
  - Download [UnixUtils](https://sourceforge.net/projects/unxutils/files/latest/download) and add the executable to path
  - Execute `patch.exe -u <path_to renderer.py>  pyrender_patch/renderer.diff`
  - Execute `patch.exe -u <path_to constants.py>  pyrender_patch/constants.diff`
  - Execute `patch.exe -u <path_to camera.py>  pyrender_patch/camera.diff`

# Download the model
We provide a pretrained model [here](https://drive.google.com/open?id=1JRQ6VQoPyQSPx3te3LX3MHIRwSYI1_fJ) for inference purposes.

# Inference
In order to run our code, a pretrained model must be present either from a training or it can be downloaded [here](#download-the-model).
Once every requirement is installed, simply rum `python inference.py [options...]`

**Important options**

`--input_path` : directory which contains depthmaps (in .pgm format)

`--output_path` : directory where results will be saved

`--scale` : multiplication factor that converts depthmap data to meters

`--saved_params_path` : path to the downloaded model

In order to see all available options with a brief description, please execute `python inference.py -h`

# Training
In order to train our model from scratch, one has to download backgrounds that are used in training time for augmentation.
**TBD: upload and add links**.
Once every requirement is installed and backgrounds are downloaded, it is time to train our model.
Execute `python main.py -h` to see all available options.


