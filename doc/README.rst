Our starting catalog is the file**, which
contains 1,376,864 objects and is the raw output of querying the [Hyperleda
database](http://leda.univ-lyon1.fr/fullsql.html) (on 2018 May 13) for all
objects with a D(25) isophotal diameter greater than 10 arcsec using the
following SQL query:

```SQL
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

**ToDo**

1. Build unWISE and GALEX mosaics.
2. Filter and sort the sample; try to remove spurious sources.
3. Include additional metadata in the webpage.
