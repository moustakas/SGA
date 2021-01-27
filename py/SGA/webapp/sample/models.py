"""
Each model will be written as a class here, 
instantiated and populated by load.py,
with each model stored as a table in the database and the fields stored as columns.
"""
from django.db.models import Model, IntegerField, CharField, FloatField, IPAddressField, DateTimeField, ManyToManyField, TextField, BooleanField

# python manage.py makemigrations SGA
# python manage.py migrate

class Sample(Model):
    """
    Model to represent a central galaxy
    """
    ra = FloatField(null=True)
    dec = FloatField(null=True)
    sga_id = IntegerField(null=True)

    # radec2xyz, for cone search in the database
    ux = FloatField(default=-2.0)
    uy = FloatField(default=-2.0)
    uz = FloatField(default=-2.0)

    group_name = CharField(max_length=40, default='')
    galaxy_name = CharField(max_length=30, default='')
    pa = FloatField(default=0.0)
    ba = FloatField(default=0.0)
    diam = FloatField(default=0.0)
    # objid = IntegerField(null=True)
    # morphtype = CharField(max_length=6, null=True)
    # ra = FloatField(null=True)
    # dec = FloatField(null=True)
    # mem_match_id = FloatField(null=True)
    # mem_match_id_string = CharField(max_length=7, null=False, primary_key = True)
    # z = FloatField(null=True)
    # la = FloatField(null=True)
    # sdss_objid = FloatField(null=True)
    # viewer_link = CharField(max_length=100, null=True)
    # skyserver_link = CharField(max_length=100, null = True)
    # 
    # def __str__(self):
    #     return ('user Central Search(%s,%s,%s,%s,%s,%s,%s,%s)' % (self.objid, self.morphtype, self.ra, self.dec, self.mem_match_id, self.z, self.la, self.sdss_objid))

    def galaxy_link(self):
        baseurl = 'https://portal.nersc.gov/project/cosmo/temp/ioannis/SGA-html-2020/'
        radeg = '%i' % int(self.ra)
        url = baseurl + radeg + '/' + self.galaxy_name + '/' + self.galaxy_name + '.html'
        return url

    def group_link(self):
        baseurl = 'https://portal.nersc.gov/project/cosmo/temp/ioannis/SGA-html-2020/'
        radeg = '%i' % int(self.ra)
        url = baseurl + radeg + '/' + self.group_name + '/' + self.group_name + '.html'
        return url

    def sga_id_string(self):
        return '%i' % self.sga_id
