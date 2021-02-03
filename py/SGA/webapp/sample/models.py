#!/usr/bin/env python

"""Each model will be written as a class here, instantiated and populated by
load.py, with each model stored as a table in the database and the fields stored
as columns.

"""
from django.db.models import (Model, IntegerField, CharField, FloatField, IPAddressField,
                              DateTimeField, ManyToManyField, TextField, BooleanField)

# python manage.py makemigrations SGA
# python manage.py migrate

class Sample(Model):
    """Model to represent a single galaxy.

    """
    sga_id = IntegerField(null=True)
    galaxy_name = CharField(max_length=30, default='')
    ra = FloatField(null=True)
    dec = FloatField(null=True)
    diam = FloatField(default=0.0)

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

    def group_link(self):
        baseurl = 'https://portal.nersc.gov/project/cosmo/temp/ioannis/SGA-html-2020/'
        #raslice = '{:06d}'.format(int(self.group_ra*1000))[:3]
        raslice = '000'
        url = baseurl + raslice + '/' + self.group_name + '/' + self.group_name + '.html'
        return url

    def sga_id_string(self):
        return '{}'.format(self.sga_id)

    def group_id_string(self):
        return '{}'.format(self.group_id)
