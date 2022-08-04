
# DistMsMatch
Distributed Semi-Supervised Multispectral Scene Classification with Few Labels

<!--
*** Based on https://github.com/othneildrew/Best-README-Template
-->



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#set-up-datasets">Set-up datasets</a></li>
        <li><a href="#run-training">Run training</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#FAQ">FAQ</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

This is the code for to realize a distributed version of [MSMatch](https://github.com/gomezzz/MSMatch). The repository includes an implementation of [FixMatch](https://arxiv.org/abs/2001.07685) for the semi-supervised training of different convolutional neural networks, including a [U-Net Encoder](https://arxiv.org/abs/1505.04597), [EfficientNet](https://arxiv.org/abs/1905.11946) and [EfficientNet Lite](https://tfhub.dev/s?deployment-format=lite&q=efficientnet%20lite) to perform scene classification on the [EuroSAT](https://github.com/phelber/EuroSAT) dataset. The code builds on and extends the [FixMatch-pytorch](https://github.com/LeeDoYup/FixMatch-pytorch) implementation based on [PyTorch](https://pytorch.org/). 
### Built With

* [PyTorch](https://pytorch.org/)
* [conda](https://docs.conda.io/en/latest/)
* [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
* [EfficientNet Lite PyTorch](https://pypi.org/project/efficientnet-lite-pytorch/)
* [albumentations](https://github.com/albumentations-team/albumentations)
* [papermill](https://papermill.readthedocs.io/en/latest/)
* imageio, numpy, pandas

<!-- GETTING STARTED -->
## Getting Started

This is a brief example of setting up DistMSMatch.

### Prerequisites

We recommend using [conda](https://docs.conda.io/en/latest/) to set-up your environment. This will also automatically set up CUDA and the cudatoolkit for you, enabling the use of GPUs for training, which is recommended.


* [conda](https://docs.conda.io/en/latest/), which will take care of all requirements for you. For a detailed list of required packages, please refer to the [conda environment file](https://github.com/gomezzz/DistMSMatch/blob/main/environment.yml).

### Installation

1. Get [miniconda](https://docs.conda.io/en/latest/miniconda.html) or similar
2. Clone the repo
   ```sh
   git clone https://github.com/gomezzz/DistMSMatch
   ```
3. Setup the environment. This will create a conda environment called `distmsmatch`
   ```sh
   conda env create -f environment.yml
   ```
4. Install [EfficientNet Lite PyTorch](https://pypi.org/project/efficientnet-lite-pytorch/)
    ```
    pip install efficientnet_lite_pytorch
    pip install efficientnet_lite0_pytorch_model
    ```

### Set up datasets
To launch the training on `EuroSAT (rgb)`, it is necessary to download the corresponding datasets. The `root_dir` variable in the `datasets/EurosatRGBDataset.py` files shall be adjusted according to the dataset path. 

### Run training
To run the training, you can exploit the `run_notebook.py` script to run a notebook. The `Baseline.ipynb` notebook contains the `MSMatch`baseline. Different configurations files in `.toml` format can be used to set-up the different `MsMatch`hyperparameters. 

To run the notebook, you can proceed as follows:

```
python run_notebook.py --notebook_path path_to_notebook --cfg_path path_to_cfg_file 
````

If `notebook_path` and `path_to_cfg_file` are not specified, the `Baseline.ipynb` notebook  will be run with default configuration.

<!-- CONTRIBUTING -->
## Contributing

The project is open to community contributions. Feel free to open an [issue](https://github.com/gomezzz/DistMSMatch/issues) or write us an email if you would like to discuss a problem or idea first.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- CONTACT -->
## Contact 

Created by ESA's [Advanced Concepts Team](https://www.esa.int/gsp/ACT/index.html) and [$\Phi$-lab](https://philab.phi.esa.int/1) (shared first authorship).

- Pablo Gómez - `pablo.gomez at esa.int` (ACT)
- Gabriele Meoni - `gabriele.meoni at esa.int` ($\Phi$-lab)

Project Link: [https://www.esa.int/gsp/ACT/projects/semisupervised/](https://www.esa.int/gsp/ACT/projects/semisupervised/)



<!-- ACKNOWLEDGEMENTS 
This README was based on https://github.com/othneildrew/Best-README-Template
-->