Siena Galaxy Atlas
==================

.. image:: https://readthedocs.org/projects/sga/badge/?version=latest
   :target: https://sga.readthedocs.io/en/latest/
   :alt: Documentation Status

The Siena Galaxy Atlas (SGA) is a multiwavelength, diameter-limited
atlas of nearby galaxies built using the `DESI Legacy Imaging
Surveys`_ (the *Legacy Surveys*). For each galaxy, the SGA delivers
precise coordinates, multiwavelength mosaics, azimuthally averaged
surface-brightness and color profiles, integrated and aperture
photometry, and additional metadata. Because large, resolved galaxies
trace the foreground extragalactic sky, the SGA also serves as a
foundational reference catalog for wide-field cosmological surveys
such as the `Dark Energy Spectroscopic Instrument (DESI)`_.

The `SGA-2020`_ was released in January 2021 and contains 383,620
galaxies selected primarily from the HyperLeda extragalactic
database. It is built on deep *grz* optical imaging from the *Legacy
Surveys* Data Release 9 (DR9) and all-sky mid-infrared imaging at
3.4–22 microns from unWISE (W1–W4).

SGA-2025 expands the atlas to nearly 500,000 galaxies across
approximately 30,000 deg² of the extragalactic sky using the *Legacy
Surveys* DR11. It adds *i*-band optical mosaics, GALEX near- and
far-ultraviolet (NUV/FUV) imaging, and a richer suite of
surface-brightness profiles and photometric measurements, making it
the most complete census of large, resolved galaxies to date.

NERSC Environment
-----------------

The SGA and SGAML Jupyter kernels set three environment variables with
NERSC defaults:

.. code-block:: bash

   SGA_DIR=/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/sga/2025
   SGA_DATA_DIR=/dvs_ro/cfs/cdirs/cosmo/data/sga/2025/data
   SGA_HTML_DIR=/dvs_ro/cfs/cdirs/cosmo/work/legacysurvey/sga/2025/html

To override any of these — or to point a kernel at a working branch —
create ``~/.sga_dev_env``; it is sourced by both kernels at startup,
after the defaults are set:

.. code-block:: bash

   # ~/.sga_dev_env — sourced by both SGA and SGAML kernels at startup
   export PATH=/global/homes/i/ioannis/code/SGA/bin/SGA2025:$PATH
   export PYTHONPATH=/global/homes/i/ioannis/code/SGA/py:$PYTHONPATH
   export SGA_DATA_DIR=/pscratch/sd/i/ioannis/SGA2025-v1.6

Any variable set here takes precedence over the kernel defaults. Delete
or empty the file to revert to the installed environment.

We gratefully acknowledge funding support for this work from the
National Science Foundation under grants AST-1616414 and AST-1909374,
and the U.S. Department of Energy, Office of Science, Office of High
Energy Physics under Award Number DE-SC0020086.

.. _`DESI Legacy Imaging Surveys`: https://legacysurvey.org
.. _`Dark Energy Spectroscopic Instrument (DESI)`: https://desi.lbl.gov
.. _`SGA-2020`: https://sga.legacysurvey.org
