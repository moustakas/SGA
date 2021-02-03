#!/usr/bin/env python

"""Load the input sample into a database table.

"""
import os
import fitsio
import django

from astropy.table import Table
from astrometry.util.starutil_numpy import radectoxyz

def main():
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SGA.webapp.settings")
    django.setup()

    from SGA.webapp.sample.models import Sample

    datafile = '/global/cfs/cdirs/cosmo/work/legacysurvey/sga/2020/SGA-2020.fits'
    columns = ['sga_id', 'ra', 'dec', 'galaxy', 'diam',
               'group_id', 'group_name', 'group_ra', 'group_dec', 'group_diameter']
    data = Table(fitsio.read(datafile, ext='SGA', columns=columns))

    print('Read {} rows from {}'.format(len(data), datafile))

    xyz = radectoxyz(data['RA'], data['DEC'])

    objs = []
    nextpow = 512 # 1024
    for i, (sgaid, gal, ra, dec, diam, gid, gname, gra, gdec, gdiam) in enumerate(zip(
        data['SGA_ID'], data['GALAXY'], data['RA'], data['DEC'], data['DIAM'],
        data['GROUP_ID'], data['GROUP_NAME'], data['GROUP_RA'], data['GROUP_DEC'], data['GROUP_DIAMETER'])):
        
        if i == nextpow:
            print('Row', i)
            nextpow *= 2
            
        sam = Sample()
        sam.row_index = i
        sam.sga_id = sgaid
        sam.galaxy_name = gal.strip()
        sam.ra = ra
        sam.dec = dec
        sam.diam = diam

        sam.ux = xyz[i, 0]
        sam.uy = xyz[i, 1]
        sam.uz = xyz[i, 2]

        sam.group_id = gid
        sam.group_name = gname.strip()
        sam.nice_group_name = gname.strip().replace('_GROUP', ' GROUP')
        sam.group_ra = gra
        sam.group_dec = gdec
        sam.group_diam = gdiam

        objs.append(sam)

    print('Bulk create')
    Sample.objects.bulk_create(objs)

if __name__ == '__main__':
    main()
