Legacy Survey Large Galaxy Atlas
================================

The Legacy Survey Large Galaxy Atlas (LSLGA) delivers multicolor images and
model fits for a diameter-limited sample of "large" galaxies in `Legacy Survey`_
imaging, which consists of deep *grz* optical imaging and unWISE *W1* and *W2*
mid-infrared imaging.

Conda Environment Setup
-----------------------

```bash
conda create --name LSLGA python=3
source activate LSLGA
conda install ipython numpy scipy matplotlib astropy jupyter pillow
conda install -c bccp nbodykit
```

.. image:: https://img.shields.io/badge/PDF-latest-orange.svg?style=flat
    :target: https://github.com/moustakas/LSLGA/blob/master-pdf/paper/ms.pdf

.. _`Legacy Survey`: http://legacysurvey.org
