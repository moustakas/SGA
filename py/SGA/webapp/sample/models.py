#!/usr/bin/env python

"""Each model will be written as a class here, instantiated and populated by
load.py, with each model stored as a table in the database and the fields stored
as columns.

"""
import os
from django.db.models import (Model, IntegerField, CharField, FloatField, IPAddressField,
                              DateTimeField, ManyToManyField, TextField, BooleanField)

# python manage.py makemigrations SGA
# python manage.py migrate

class Sample(Model):
    """Model to represent a single galaxy.

    """
    # in FITS table
    row_index = IntegerField(default=-1)

    sga_id = IntegerField(null=True)
    galaxy_name = CharField(max_length=30, default='')
    ra = FloatField(null=True)
    dec = FloatField(null=True)
    diam = FloatField(default=0.0)

    group_primary = BooleanField(default=False)
    group_id = IntegerField(null=True)
    group_name = CharField(max_length=40, default='')
    group_ra = FloatField(null=True)
    group_dec = FloatField(null=True)
    group_diam = FloatField(default=0.0)

    # radec2xyz, for cone search in the database
    ux = FloatField(default=-2.0)
    uy = FloatField(default=-2.0)
    uz = FloatField(default=-2.0)

    #def galaxy_link(self):
    #    radeg = '%i' % int(self.ra)
    #    url = baseurl + radeg + '/' + self.galaxy_name + '/' + self.galaxy_name + '.html'
    #    return url

    #def group_link(self):
    #    baseurl = 'https://portal.nersc.gov/project/cosmo/temp/ioannis/SGA-html-2020/'
    #    #raslice = '{:06d}'.format(int(self.group_ra*1000))[:3]
    #    raslice = '000'
    #    url = baseurl + raslice + '/' + self.group_name + '/' + self.group_name + '.html'
    #    return url

    def base_html_dir(self):
        return '/global/cfs/cdirs/cosmo/data/sga/2020/html/'

    def png_base_url(self):
        baseurl = 'https://portal.nersc.gov/project/cosmo/data/sga/2020/html/'
        baseurl += self.ra_slice() + '/' + self.group_name + '/';
        return baseurl

    def mosaic_diam(self):
        if self.group_diam > 30: # NGC0598=M33 is 61 arcmin in diameter!
            mosaic_diam = self.group_diam * 2 * 0.7 # [arcmin]
        elif self.group_diam > 14 and self.group_diam < 30:
            mosaic_diam = self.group_diam * 2 * 1.0 # [arcmin]
        else:
            mosaic_diam = self.group_diam * 2 * 1.5 # [arcmin]
        return '{:.3f}'.format(mosaic_diam) # [arcmin]

    def ra_slice(self):
        raslice = '{:06d}'.format(int(self.group_ra*1000))[:3]
        return raslice

    def sga_id_string(self):
        return '{}'.format(self.sga_id)

    def group_ra_string(self):
        return '{:.7f}'.format(self.group_ra)

    def group_dec_string(self):
        return '{:.7f}'.format(self.group_dec)

    def ra_string(self):
        return '{:.7f}'.format(self.ra)

    def dec_string(self):
        return '{:.7f}'.format(self.dec)

    def group_id_string(self):
        return '{}'.format(self.group_id)

    def ellipsefile(self):
        ellipsefile = '{}{}-largegalaxy-{}-ellipse-sbprofile.png'.format(self.png_base_url(), self.group_name, self.sga_id_string())
        return ellipsefile
    
    def ellipse_exists(self):
        ellipsefile = os.path.join(self.base_html_dir(), self.ra_slice(), self.group_name, '{}-largegalaxy-{}-ellipse-sbprofile.png'.format(
            self.group_name, self.sga_id_string()))
        return os.path.isfile(ellipsefile)
