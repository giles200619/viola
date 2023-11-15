# viola


### Submodules:
#### Open3d
At the time when this project was created, we use the RGBD reconstruction pipeline from [Open3d](https://github.com/isl-org/Open3D) v0.17.0<br />
Clone the Open3d project and modify the `--open3d_path` in `./preprocess/redwood_open3d_m2f.py` accordingly
#### Mask2Former
Download Mask2Former model weight from their [Model zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md), in this project we are using model id: 47429163_0<br />
The path to the model weight should look like:<br />
`./mask2former/model_weights/model_final_47429163_0.pkl`<br />

#### Simplerecon
Download the `hero_model` weight from [Simplerecon](https://github.com/nianticlabs/simplerecon#-models)<br />
Path to the weight should look like:<br />
`./simplerecon/weights/hero_model.ckpt`<br />

