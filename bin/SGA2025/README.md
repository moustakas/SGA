# Siena Galaxy Atlas 2025

## Gather External Catalogs



## Build the Parent Sample

```bash
SGA2025-build-parent --build-parent-nocuts
SGA2025-build-parent --build-parent-vicuts
SGA2025-build-parent --build-parent-archive
```

Find objects in the footprint:
```bash
SGA2025-build-parent --in-footprint --region=dr9-north
SGA2025-build-parent --in-footprint --region=dr11-south
SGA2025-build-parent --build-parent
```

Build QA:
```bash
SGA2025-build-parent --qa-parent
```

For investigating early samples, a very powerful wrapper for
generating cutouts was developed called `SGA2025-cutouts`. Here are a
few options:

By default `SGA2025-cutouts` will generate, well, cutouts. So, e.g.,
```
SGA2025-cutouts --region=dr9-north --ntest=16 --mp=4


```
