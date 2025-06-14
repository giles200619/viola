## Environment setup

We recommend creating a new virtual environment. After activating the virtual environment, install pytorch, pytorch3d, lightning and other required packages:

```
conda create -n viola python=3.10
conda activate viola
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
conda install lightning -c conda-forge
conda install -c conda-forge pycocotools cudatoolkit-dev
pip install -r requirements.txt
```


### Thirdparty modules:
#### Open3D
VioLA uses the RGBD reconstruction pipeline from [Open3D](https://github.com/isl-org/Open3D) <br />
Clone the Open3D repository and modify the `--open3d_path` in `preprocess/redwood_open3d_m2f.py` accordingly
```
git clone https://github.com/isl-org/Open3D.git
```

#### Mask2Former
Download Mask2Former model weight from their [Model zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md), in this project we are using model id: 47429163_0<br />
```
mkdir mask2former/model_weights
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl -O mask2former/model_weights/model_final_47429163_0.pkl
```
The path to the model weight should look like:<br />
`viola/mask2former/model_weights/model_final_47429163_0.pkl`<br />
The Mask2Former model uses Detectron2, to install it from source, [run](https://detectron2.readthedocs.io/en/latest/tutorials/install.html):
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

cd mask2former/mask2former/modeling/pixel_decoder/ops
sh make.sh
```

#### SimpleRecon

VioLA uses SimpleRecon for reconstructing posed RGB videos.

Download the `hero_model` weight from [SimpleRecon](https://github.com/nianticlabs/simplerecon#-models). You can download the file from terminal using gdown:<br />
```
pip install gdown
gdown https://drive.google.com/uc?id=1hCuKZjEq-AghrYAmFxJs_4eeixIlP488
mv hero_model.ckpt simplerecon/weights/
```
Path to the weight should look like:<br />
`viola/simplerecon/weights/hero_model.ckpt`<br />
Due to torch_lightning version difference, you might encouter missing key error while loading `hero_model.ckpt`. We provide a script to add missing keys to the checkpoint:<br />
```
python ./utils/fix_hero_model_keys.py 
```

### Scene completion (optional)

For using the scene completion module, we need additional packages. 

Install the pointersect model used for rendering the reconstructed point cloud from target views:
```
pip install pointersect
```

Download the model weights for the depth estimation model using the instructions in [the IronDepth repository](https://github.com/baegwangbin/IronDepth).<br />
- Download the files `normal_scannet.pt` and `irondepth_scannet.pt`
- Move them to the directory `viola/checkpoints`


There is a version mismatch for the OpenEXR package. To fix this issue, run:
```
pip uninstall openexr
conda install -c conda-forge openexr-python
```
