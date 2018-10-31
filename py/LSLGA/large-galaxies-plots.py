#!/usr/bin/env python

"""
Generate figures for the large-galaxies TechNote.

J. Moustakas
Siena College
jmoustakas@siena.edu

2016 July 01

make some plots comparing with NSA -- redshifts, coordinates, bmag
bmag (or imag) vs diameter (+marginalized distributions)

"""

from __future__ import division, print_function

import os
import sys
import pdb
import argparse

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from astropy.io import fits

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dr', type=str, default='dr2', help='DECaLS Data Release')
    parser.add_argument('--radec', action='store_true', help='Dec vs RA')
    parser.add_argument('--d25-bmag', action='store_true', help='D(25) vs Bmag')

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # Top-level directory
    key = 'LEGACY_SURVEY_LARGE_GALAXIES'
    if key not in os.environ:
        print('Required ${} environment variable not set'.format(key))
        return 0
    largedir = os.getenv(key)

    # TechNote directory
    qadir = os.path.join(os.getenv('DESIDOCS'), 'technotes', 'imaging',
                         'large-galaxies', 'trunk', 'figures')

   # Set sns preferences.  Pallet choices: deep, muted, bright, pastel, dark, colorblind
    sns.set(style='white', font_scale=1.5, font='sans-serif', palette='Set2')
    setcolors = sns.color_palette()

    dr = args.dr.lower()
    samplefile = os.path.join(largedir, 'sample', 'large-galaxies-{}.fits'.format(dr))
    print('Reading {}'.format(samplefile))
    sample = fits.getdata(samplefile, 1)

    parentfile = os.path.join(largedir, 'sample', 'leda-logd25-0.05.fits.gz')
    print('Reading {}'.format(parentfile))
    parent = fits.getdata(parentfile)

    # Figure: dec vs ra
    if args.radec:
        qafile = os.path.join(qadir, 'lslga-{}-radec.pdf'.format(dr))
        fig, ax = plt.subplots(figsize=(8, 5))

        im = ax.hexbin(parent['ra'], parent['dec'], bins='log', cmap=plt.cm.Blues_r,
                       mincnt=1, label='Parent Sample', extent=(0, 360, -90, 90))
        #cb = fig.colorbar(im, ticks=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        cb = fig.colorbar(im, ticks=[np.log10([3, 10, 30, 100, 300, 1000])])
        cb.ax.set_yticklabels(['3', '10', '30', '100', '300', '1000'])
        cb.set_label('log$_{10}$ (Number of Galaxies per Bin)')

        #ax.scatter(parent['ra'], parent['dec'], marker='s', s=2, color=setcolors[3],
        #           label='Parent Sample')
        ax.scatter(sample['ra'], sample['dec'], marker='s', s=3, color='tomato',
                   label='LSLGA-DR2')
        ax.set_xlabel('Right Ascension (degrees)')
        ax.set_xlim((0, 360))
        ax.set_ylim((-90, 90))
        #ax.text(0.85, 0.85, '{}-band'.format(filt), ha='center', va='bottom',
        #                transform=thisax.transAxes, fontsize=18, color=col)
        ax.set_ylabel('Declination (degrees)')
        #ax.legend()

        plt.subplots_adjust(bottom=0.2)
        print('Writing {}'.format(qafile))
        plt.savefig(qafile)
    
    # Figure: D(25) vs Bmag
    if args.d25_bmag:
        qafile = os.path.join(qadir, 'lslga-{}-d25-bmag.pdf'.format(dr))
        fig, ax = plt.subplots()

        extent = (0, 24, -1.0, 3.0)

        # Plot the parent sample.
        good = np.where(parent['bmag'] > -900)[0]
        bmag = parent['bmag'][good].flatten()
        d25 = np.log10(parent['d25'][good].flatten()/60.0)

        m31 = np.where(np.char.strip(parent['galaxy'][good]) == 'NGC0224')[0][0]
        ax.annotate('M31', xy=(bmag[m31], d25[m31]), xytext=[bmag[m31]+3, d25[m31]+0.4], 
                    arrowprops=dict(facecolor='gray', shrink=0.1, width=1, 
                                    headwidth=6), fontsize=12, horizontalalignment='center',
                                    verticalalignment='top')

        smc = np.where(np.char.strip(parent['galaxy'][good]) == 'NGC0292')[0][0]
        ax.annotate('SMC', xy=(bmag[smc], d25[smc]), xytext=[bmag[smc]-1.5, d25[smc]-0.8], 
                    arrowprops=dict(facecolor='gray', shrink=0.1, width=1, 
                                    headwidth=6), fontsize=12, horizontalalignment='center',
                                    verticalalignment='top')
        
        ax.hexbin(bmag, d25, bins='log', extent=extent, mincnt=1, cmap=plt.cm.Blues_r)
        
        # Now plot our sample.
        good = np.where(sample['bmag'] > -900)[0]
        bmag = sample['bmag'][good].flatten()
        d25 = np.log10(sample['radius'][good].flatten()*2.0/60.0)
        ax.scatter(bmag, d25, marker='o', color='tomato', s=5, edgecolor='gray')
        ax.set_xlabel('B-band (Vega) magnitude')
        ax.set_ylabel(r'log$_{10}$ D(25) (arcmin)')
        ax.set_xlim(extent[0:2])
        ax.set_ylim(extent[2:4])

        d25min, d25max = (0.5, 10.0) # see large-galaxies.py [arcmin]
        ax.hlines(np.log10(d25min), extent[0], extent[1], colors='k', linestyles='dashed')
        ax.hlines(np.log10(d25max), extent[0], extent[1], colors='k', linestyles='dashed')
             
        plt.subplots_adjust(bottom=0.2)
        print('Writing {}'.format(qafile))
        plt.savefig(qafile)
    
if __name__ == '__main__':
    main()
