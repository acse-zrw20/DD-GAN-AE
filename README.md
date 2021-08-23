# DD-GAN-AE

## Autoencoders for the DD-GAN

[![codecov](https://codecov.io/gh/acse-zrw20/DD-GAN-AE/branch/main/graph/badge.svg?token=1LU7UG5OF9)](https://codecov.io/gh/acse-zrw20/DD-GAN-AE)
[![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/acse-zrw20/DD-GAN-AE/blob/main/LICENSE)
[![Documentation Status](https://github.com/acse-zrw20/DD-GAN-AE/actions/workflows/docs.yml/badge.svg)](https://github.com/acse-zrw220/DD-GAN-AE/blob/main/docs/docs.pdf)
![example workflow](https://github.com/acse-zrw20/DD-GAN-AE/actions/workflows/health.yml/badge.svg)

<!-- PROJECT LOGO -->

<br />
<p align="center">
  <a href="https://github.com/acse-zrw20/DD-GAN-AE">
    <img src="images/dataflow.png" alt="Logo" width="1000" height="600">
  </a>

<p align="center">
    <br />
    <a href="https://github.com/acse-zrw20/DD-GAN-AE/blob/main/docs/docs.pdf"><strong>Explore the docs»</strong></a>
    <br />
    <br />
    <a href="https://github.com/acse-zrw20/DD-GAN-AE/issues">Report Bug</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This project contains an intuitive library for interacting with compressive and predictive methods for predicting fluid dynamics simulations efficiently. Code was built for a Msc thesis at the Imperial College London.

Read the [documentation](https://github.com/acse-zrw20/DD-GAN-AE/blob/main/docs/docs.pdf) for further info!

<!-- GETTING STARTED -->

- `ddganAE` contains the main source code of the developed package with four subfolders: 
  - `architectures` functions as a library of model architectures
  - `wandb` contains some subroutines for hyperparameter-optimizing the developed models with wandb
  - `models` contains the main logic that the implemented models use, everything except for the architecture
  - `preprocessing` contains any preprocessing subroutines included in the package.
- `docs` contains the accompanying documentation.
- `examples` contains example notebooks. Links to their Colab versions which can be readily executed are also provided below.
- `hpc` contains bash scripts for interacting with Imperial College London's Research Computing Service (high performance computer).
- `images` contains any accompanying images.
- `preprocessing` contains some preprocessing functions specific to the data used in this research and thus not included in the main package.
- `submodules` contains any relevant submodules.
- `tests` contains any tests written for the produced package.

## Prerequisites

* Python 3.8
* Tensorflow and other packages in ```requirements.txt```
* (Optional) GPU with CUDA

## Installation

Follow these steps to install:

1. ```git clone https://github.com/acse-zrw20/DD-GAN-AE```
2. ```cd ./DD-GAN-AE```
3. ```pip install -e .```

<!-- USAGE EXAMPLES -->

## Getting Started

In a python file, import the following to use all of the functionality:

```python
import ddganAE
```
Training a model for reconstruction:

```python
from ddganAE.models import CAE
from ddganAE.architectures.cae.D2 import *
import tf

input_shape = (55, 42, 2)
dataset = np.load(...) # dataset with shape (<nsamples>, 55, 42, 2)

optimizer = tf.keras.optimizers.Adam() # Define an optimizer
initializer = tf.keras.initializers.RandomNormal() # Define a weights initializer

# Define any encoder and decoder, see docs for more premade architectures
encoder, decoder = build_omata_encoder_decoder(input_shape, 10, initializer)

cae = CAE(encoder, decoder, optimizer) # define the model
cae.compile(input_shape) # compile the model

cae.train(dataset, 200) # train the model with 200 epochs

recon_dataset = cae.predict(dataset) # pass the dataset through the model and generate outputs
```

Training a model for prediction:

```python
from ddganAE.models import Predictive_adversarial
from ddganAE.architectures.svdae import *
import tf

latent_vars = 100  # Define the number of variables the predictive model will use in discriminator layer
n_predicted_vars = 10 # Define the number of predicted variables

dataset = np.load(...) # dataset with shape (<ndomains>, 10, <ntimesteps>)

optimizer = tf.keras.optimizers.Adam() # Define an optimizer
initializer = tf.keras.initializers.RandomNormal() # Define a weights initializer

# Define any encoder and decoder, see docs for more premade architectures. Note for predictive
# models we don't necessarily need to use encoders or decoders
encoder = build_slimmer_dense_encoder(latent_vars, initializer)
decoder = build_slimmer_dense_decoder(n_predicted_vars, latent_vars, initializer)
discriminator = build_custom_discriminator(latent_vars, initializer)

pred_adv = Predictive_adversarial(encoder, decoder, discriminator, optimizer)
pred_adv.compile(n_predicted_vars, increment=False)
pred_adv.train(dataset, 200)

# Select the boundaries with all timesteps
boundaries = np.zeros((2, 10, <ntimesteps>))
boundaries[0], boundaries[1]  = dataset[2], dataset[9] # third and 10th subdomains used as boundaries

# Select the initial values at the first timestep
init_values = val_data[3:9, :, 0]

predicted_latent = pred_adv.predict(boundaries, init_values, 50, # Predict 50 steps forward 
                                    iters=4, sor=1, pre_interval=False)
```

## Examples

* Compression usage examples on flow past cylinder dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oxLf-SayXWrG_grniEptbmIwhMo4XMCD#offline=true&sandboxMode=true)
* Compression usage examples on slug flow dataset [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hsRsPp64dbQz0f3zG7nwcENvSKoCScoM#offline=true&sandboxMode=true)

These notebooks can also be found under examples in this repository
<!-- ACKNOWLEDGEMENTS 
_For more information, please refer to the report in this repo_
-->
<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->


## Contact

* Wolffs, Zef zefwolffs@gmail.com

<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

* Dr. Claire Heaney
* Prof. Christopher Pain
* Royal School of Mines, Imperial College London

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links 
[contributors-shield]: https://img.shields.io/github/contributors/acse-zrw20/group-project-the-uploaders.svg?style=for-the-badge
[contributors-url]: https://github.com/acse-zrw20/DD-GAN-AE/graphs/contributors
[issues-shield]: https://img.shields.io/github/issues/acse-2020/group-project-the-uploaders.svg?style=for-the-badge
[issues-url]: https://github.com/acse-2020/acse-4-x-ray-classification-losslandscape/issues
[license-shield]: https://img.shields.io/github/license/acse-2020/group-project-the-uploaders.svg?style=for-the-badge
[license-url]: https://github.com/acse-2020/acse-4-x-ray-classification-losslandscape/blob/main/LICENSE.txt
-->
