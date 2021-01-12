"""
Load and parse through an input data file. Currently using "centrals-sample.fits".
Creates and populates a database table to represent centrals model
"""
import os,sys
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "SGA.webapp.settings")
import django
django.setup()

from SGA.webapp.sample.models import Sample
import fitsio
from astropy.table import Table

#if True:
## csv_filepathname="centrals-sample.csv"
## dataReader = csv.reader(open(csv_filepathname), delimiter=',', quotechar='"')
## for row in dataReader:
##     print('Row:', row)
##     centrals=Centrals()
##     centrals.objid=row[0]
##     centrals.morphtype=row[1]
##     centrals.ra = row[2]
##     centrals.dec = row[3]
##     centrals.mem_match_id = row[4]
##     centrals.z = row[5]
##     centrals.la = row[6]
##     centrals.sdss_objid = row[7]
##     centrals.save()

#
data = Table(fitsio.read('/global/cfs/cdirs/cosmo/work/legacysurvey/SGA-2020/SGA-ellipse-v3.2.fits', columns=['sga_id','ra','dec']))
#data.rename_column('type', 'morphtype')
print('Read', len(data), 'rows')
data = data[data['SGA_ID'] >= 0]
print('Cut to', len(data), 'with SGA_ID')

objs = []
nextpow = 1024
for i,(sgaid,ra,dec) in enumerate(zip(data['SGA_ID'], data['RA'], data['DEC'])):
    if i == nextpow:
        print('Row', i)
        nextpow *= 2
    sam=Sample()
    sam.sga_id = sgaid
    sam.ra     = ra
    sam.dec    = dec
    objs.append(sam)
    #sam.save()

print('Bulk create')
Sample.objects.bulk_create(objs)
    
# for row in data:
#     #print('Row:', row)
#     sam=Sample()
#     sam.sga_id = row[0]
#     sam.ra     = row[1]
#     sam.dec    = row[2]
#     sam.save()

## for row in data:
##     centrals = Centrals()
##     for col in data.colnames:
##         col.getattr(centrals, col) = data[row][col]
##     centrals.save()
