#!/usr/bin/env python

"""Load the input sample into a database table.

"""
import os
import numpy as np
import fitsio
import django

from astropy.table import Table, hstack
from astrometry.util.starutil_numpy import radectoxyz

def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SGA.webapp.settings")
    django.setup()

    from SGA.webapp.sample.models import Sample

    datadir = '/global/cfs/cdirs/cosmo/work/legacysurvey/sga/2020'
    sgafile = os.path.join(datadir, 'SGA-2020-ls.fits')

    sga_columns = ['sga_id', 'galaxy', 'morphtype', 'ra_leda', 'dec_leda', 'd25_leda', 'pa_leda', 'ba_leda',
                   'diam', 'pa', 'ba', #'majoraxis',
                   'group_id', 'group_name', 'group_ra', 'group_dec', 'group_diameter', 'group_primary',
                   'g_mag_sb24', 'g_mag_sb24', 'g_mag_sb24', 'r_mag_sb24', 'r_mag_sb24', 'r_mag_sb24', 'z_mag_sb24', 'z_mag_sb24', 'z_mag_sb24', 
                   'g_mag_sb25', 'g_mag_sb25', 'g_mag_sb25', 'r_mag_sb25', 'r_mag_sb25', 'r_mag_sb25', 'z_mag_sb25', 'z_mag_sb25', 'z_mag_sb25', 
                   'g_mag_sb26', 'g_mag_sb26', 'g_mag_sb26', 'r_mag_sb26', 'r_mag_sb26', 'r_mag_sb26', 'z_mag_sb26', 'z_mag_sb26', 'z_mag_sb26',
                   'radius_sb24', 'radius_sb25', 'radius_sb26'
                   ]
                   
    tractor_cols = ['type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
                    'flux_g', 'flux_r', 'flux_z', 'flux_ivar_g', 'flux_ivar_r', 'flux_ivar_z']
       
    sga = Table(fitsio.read(sgafile, ext='SGA-LS', columns=sga_columns))
    sga_tractor = Table(fitsio.read(sgafile, ext='SGA-TRACTOR', columns=tractor_cols))
    sga = hstack((sga, sga_tractor))
    print('Read {} rows from {}'.format(len(sga), sgafile))

    xyz = radectoxyz(sga['RA_LEDA'], sga['DEC_LEDA'])

    objs = []
    nextpow = 1024
    for ii, onegal in enumerate(sga):
        if ii == nextpow:
            print('Row', ii)
            nextpow *= 2

        sam = Sample()
        sam.row_index = ii
        sam.ux = xyz[ii, 0]
        sam.uy = xyz[ii, 1]
        sam.uz = xyz[ii, 2]

        for col in sga.colnames:
            val = onegal[col]
            if type(val) == np.str or type(val) == np.str_:
                val.strip()

            setattr(sam, col.lower(), val)
            if col == 'GROUP_NAME':
                setattr(sam, 'nice_group_name', val.replace('_GROUP', ' Group'))

        objs.append(sam)
            
    print('Bulk creating the database.')
    Sample.objects.bulk_create(objs)

if __name__ == '__main__':
    main()
