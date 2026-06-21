Installation
============

Requirements
------------

SGA requires Python 3.9 or later and the following packages:
`numpy <https://numpy.org>`_,
`astropy <https://www.astropy.org>`_,
`fitsio <https://github.com/esheldon/fitsio>`_,
`matplotlib <https://matplotlib.org>`_,
`scipy <https://scipy.org>`_,
`photutils <https://photutils.readthedocs.io>`_, and
`pydl <https://pydl.readthedocs.io>`_.

Basic Install
-------------

Install the latest release from PyPI::

   pip install SGA

Or install directly from the GitHub repository::

   pip install git+https://github.com/moustakas/SGA.git

Development Install
-------------------

Clone the repository and install in editable mode::

   git clone https://github.com/moustakas/SGA.git
   cd SGA
   pip install --no-deps -e .

To also install the dependencies needed to build this documentation locally::

   pip install -e ".[doc]"
   sphinx-build doc doc/_build/html

Full Environment (NERSC or Laptop)
-----------------------------------

The ``etc/`` directory contains conda environment specs and setup scripts that
build the full dependency stack — including ``astrometry.net`` (from source),
``tractor``, ``legacypipe``, ``pydl``, and SGA — into a shared conda
environment. See ``etc/README.md`` for complete instructions.

.. code-block:: bash

   # NERSC
   module load conda
   bash etc/create-env.sh

   # Laptop (requires micromamba)
   bash etc/create-env-laptop.sh

The shared NERSC environment lives at
``/global/common/software/desi/users/ioannis/SGA``.
