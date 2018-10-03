Guide to *LSLGA* Analysis
=========================

This README briefly describes our analysis procedure, and the purpose and
content of each notebook.

Input Data and Sample Selection
-------------------------------

1. *Query Hyperleda for an angular-diameter limited sample of galaxies.*

   First, construct a super-sample of all known galaxies (N=1,376,864 as of 2018
   May 13) with isophotal D(25) angular diameters greater than 10 arcsec by
   executing the following SQL query using `Hyperleda`_:
   
   ```
    SELECT
    pgc, objname, objtype, al2000, de2000, type, bar, ring,  
    multiple, compactness, t, logd25, logr25, pa, bt, it,  
    kt, v, modbest
    WHERE
    logd25 > 0.2218487 and (objtype='G' or objtype='M' or objtype='M2' or  
                           objtype='M3' or objtype='MG' or objtype='MC')
    ORDER BY
    al2000
   ```

   resulting in the catalog *hyperleda-d25min10-18may13.txt*.

2. *Parse and supplement the Hyperleda catalog with AllWISE photometry.*

   Next, parse the raw `Hyperleda`_ output into a FITS file using the script
   `LSLGA-parse-hyperleda`_, and then produce a row-matched `AllWISE`_ catalog,
   *hyperleda-d25min10-18may13-allwise.fits*, by submitting the SLURM script
   `match-hyperleda-allwise.slurm`_ at NERSC.

3. *Create the parent sample of large galaxies.*

   Build the parent sample of large galaxies using the script
   `LSLGA-build-parent`_.  This sample is defined as the set of galaxies in the
   full `Hyperleda`_ catalog with a finite magnitude estimate.  (Visually
   investigating the objects without magnitude estimates reveals that the
   overwhelming majority are spurious.)

   There is also an option here to apply minimum and maximum angular diameter
   cuts, but in detail we do not apply any.

   In addition, to flag galaxies near bright stars; a particularly nice example
   (that we do not want to throw out!) is `IC2204`_, a beautiful r=13.4 disk
   galaxy at a redshift of z=0.0155 with an angular diameter of approximately
   1.1 arcmin.

   Finally, we identify galaxies within the (current) DESI footprint, but do not
   explicitly restrict the sample to be within this footprint.

4. *Generate the sample of galaxies with DR6 or DR7 imaging.*

   Finally, we 


.. _`Hyperleda`: http://leda.univ-lyon1.fr/fullsql.html

.. _`LSLGA-parse-hyperleda`: https://github.com/moustakas/LSLGA/blob/master/bin/LSLGA-parse-hyperleda

.. _`match-hyperleda-allwise.slurm`: https://github.com/moustakas/LSLGA/blob/master/bin/match-hyperleda-allwise.slurm

.. _`AllWISE`: http://wise2.ipac.caltech.edu/docs/release/allwise/

.. _`LSLGA-build-parent`: https://github.com/moustakas/LSLGA/blob/master/bin/LSLGA-build-parent

.. _`IC2204`_: http://legacysurvey.org/viewer?ra=115.3331&dec=34.2240&zoom=12&layer=mzls+bass-dr6

Analysis for Paper 1
--------------------

Write me!


Future Work
-----------

What's next?


References
----------

**Relevant papers**
