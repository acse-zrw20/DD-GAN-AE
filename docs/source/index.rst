.. DD-GAN-AE documentation master file, created by
   sphinx-quickstart on Sat Jun  5 17:55:26 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DD-GAN's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Models
==================

This package contains three models that can readily be used with user defined 
or imported network architectures. These consist of: adversarial autoencoder, 
convolutional autoencoder, and SVD autoencoder. 

Adversarial Autoencoder
-----------------------
.. automodule:: models.aae
   :members:
   :undoc-members:

Convolutional Autoencoder
--------------------------
.. automodule:: models.cae
   :members:
   :undoc-members:

SVD Autoencoder
--------------------------
.. automodule:: models.svdae
   :members:
   :undoc-members:

Hyperparameter optimization
===========================

This package contains some functionality for doing hyperparameter optimization
with the Weights and Biases platform. Below is the documentation for the
functions that handle this for the flow past cylinder and slug flow problems.

Flow Past Cylinder
--------------------------
.. automodule:: wandb.train_wandb_fpc
   :members:
   :undoc-members:

Slug Flow
--------------------------
.. automodule:: wandb.train_wandb_sf
   :members:
   :undoc-members:

Utilities
===========================

This package also contains some utilities for printing, loss 
functions, etc...

.. automodule:: utils
   :members:
   :undoc-members:

.. automodule:: preprocessing.utils
   :members:
   :undoc-members:

Library of Architectures
===========================

While the package is built in such a way that the user can easily use
the architectures they designed. This package also includes a set of premade
architectures. These are listed below.


Convolutional Architectures
-----------------------------

Note that we have different architectures for the flow past cylinder and slug
flow problems.

Two Dimensional (Flow Past Cylinder)
*************************************
.. automodule:: architectures.cae.D2.cae
   :members:
   :undoc-members:

Three Dimensional (Slug Flow)
*************************************
.. automodule:: architectures.cae.D3.cae
   :members:
   :undoc-members:

Discriminator Architectures
-----------------------------

.. automodule:: architectures.discriminators.discriminators
   :members:
   :undoc-members:

Mixed architectures for SVD Autoencoder
----------------------------------------

.. automodule:: architectures.svdae.svdae
   :members:
   :undoc-members:
