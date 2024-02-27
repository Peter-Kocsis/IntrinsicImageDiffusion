<p align="center">

  <h1 align="center">Intrinsic Image Diffusion for Single-view Material Estimation</h1>
  <p align="center">
    <a href="https://peter-kocsis.github.io/">Peter Kocsis</a>
    ·
    <a href="https://www.vincentsitzmann.com/">Vincent Sitzmann</a>
    ·
    <a href="https://niessnerlab.org/members/matthias_niessner/profile.html">Matthias Niessner</a>
  </p>
  <h2 align="center">CVPR 2024</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2312.12274">Paper</a> | <a href="https://peter-kocsis.github.io/IntrinsicImageDiffusion/">Project Page</a> </h3>
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="./res/iid.gif" alt="Logo" width="95%">
  </a>
</p>

<p align="center">
We utilize the strong prior of diffusion models and formulate the material estimation task probabilistically. Our approach generates multiple solutions for a single input view, with much more details and sharper features compared to previous works. 
</p>
<br>

## Structure
Our project has the following structure:

```
├── docs                  <- Project page
├── iid                   <- Our main package for Intrinsic Image Diffusion
│   ├── ldm                  <- Stable Diffusion related code
│   ├── iid.py               <- Our main module
│   ├── transform.py         <- Data transforms
│   └── utils.py             <- Utility functions and classes
├── models                 <- Model and config folder
├── res                    <- Documentation resources
├── environment.yaml       <- Env file for creating conda environment
└── README.md
```

# Installation
To install the dependencies, you can use the provided environment file:

```
conda env create -n iid -f environment.yml
```

#### (Optional) XFormers
For better performance, installing [XFormers](https://github.com/facebookresearch/xformers) is recommended.
```
conda install xformers -c xformers
```

### Model
Download the model to the `models` folder.
```
wget "https://syncandshare.lrz.de/dl/fiAomi6K8g5dywJBwAxFiZ/iid_e250.pth" -O "models/iid_e250.pth"
```

# Material Diffusion Training
Coming soon!

# Material Diffusion Inference
Running the model requires at least 10GB of GPU memory.
Code for inference can be found in our main module. 
This script will load the test image from the `res` folder, predicts a single material explanation and saves it to the same folder. 
You can run it with the following command:
```
python -m iid --input res/test.png --output output/test_out.png 
```
By default, the script predicts 10 material explanations and computes the average. 

 <div  class="row">
  <img src="res/test.png"/> 
  <img src="res/test_out.png"/>
  <img src="res/test_out_roughness.png"/>
  <img src="res/test_out_metal.png"/>
</div>

# Lighting Optimization
Coming soon!

# Acknowledgements
This project is built upon [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion), we are grateful for the authors open-sourcing their project. 
We used [Hydra](https://github.com/facebookresearch/hydra) configuration management with [Pythorch Lightning](https://github.com/Lightning-AI/pytorch-lightning). 
Our model was trained on the high-quality [InteriorVerse](https://interiorverse.github.io/) synthetic indoor dataset. 
Rendering model was inspired by [Zhu et. al. 2022](https://github.com/jingsenzhu/IndoorInverseRendering). 

# Citation
If you find our code or paper useful, please cite
```bibtex
@article{Kocsis2024IID,
  author    = {Kocsis, Peter and Sitzmann, Vincent and Nie\{ss}ner, Matthias},
  title     = {Intrinsic Image Diffusion for Single-view Material Estimation},
  journal   = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024},
}
```