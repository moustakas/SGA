#!/usr/bin/env python

def main():
    import os
    import numpy as np
    import fitsio

    import astropy.table
    
    lslgadir = '/Users/ioannis/research/projects/LSLGA/sample/v2.0'
    lslgafile = os.path.join(lslgadir, 'LSLGA-v2.0.fits')
    lslga = astropy.table.Table.read(lslgafile)
    print('Read {} galaxies from {}'.format(len(lslga), lslgafile))

    lslga = lslga['IN_DESI'][:10]

    base = 'http://legacysurvey.org/viewer-dev/jpeg-cutout?'

    manifestfile = 'lslga-manifest.csv'
    print('Writing {}'.format(manifestfile))
    with open(manifestfile, 'w') as manifest:
        for gal in lslga:
            size = np.round(gal['D25'] * 1.5 * 60 / 0.262).astype(int)
            for ii, imtype in enumerate(('', '-model', '-resid')):
                jpgfile.append('jpg/{}{}.jpg'.format(gal['GALAXY'].lower(), imtype))
                url = '"{}ra={}&dec={}&size={}&layer=dr8{}"'.format(
                    base, gal['RA'], gal['DEC'], size, imtype)
            manifest.write()


if __name__ == '__main__':
    main()
