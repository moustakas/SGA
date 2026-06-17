#!/usr/bin/env python
"""
checksums_sga2025.py — Generate sha256 checksums for SGA2025 group directories.

Usage (single-process):
    python checksums_sga2025.py /path/to/dr11-south

Usage (MPI):
    srun -n 64 python checksums_sga2025.py /path/to/dr11-south

For each group directory, writes:
    sga_2025_data_{region}_{raslice}_{group_name}.sha256sum

Skips directories where the checksum file already exists.

Directory structure assumed:
    {topdir}/           <- e.g. dr11-south  (= {region})
      {raslice}/        <- e.g. 000
        {group_name}/   <- e.g. 00000m0649

salloc -N 1 -C cpu -A desi -t 01:00:00 --qos interactive
source /dvs_ro/common/software/desi/desi_environment.sh main
time srun --ntasks=64 python checksums_sga2025.py /pscratch/sd/i/ioannis/SGA2025-v1.6/dr11-north --logfile checksums-dr11-north.log
time srun --ntasks=64 python checksums_sga2025.py /pscratch/sd/i/ioannis/SGA2025-v1.6/dr11-south --logfile checksums-dr11-south.log
"""

import sys
import hashlib
import argparse
import logging
from glob import glob
from pathlib import Path

# ---------------------------------------------------------------------------
# MPI setup (graceful fallback to single-process)
# ---------------------------------------------------------------------------

try:
    from mpi4py import MPI
    _comm = MPI.COMM_WORLD
    _rank = _comm.Get_rank()
    _size = _comm.Get_size()
    _use_mpi = _size > 1
except ImportError:
    _comm, _rank, _size, _use_mpi = None, 0, 1, False


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------

def sha256_file(filepath):
    """Return hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def process_group(dirpath, region, raslice, group_name):
    """
    Write a sha256sum file for all files in dirpath.

    Returns 'skipped' if the checksum file already exists, 'done' otherwise.
    """
    cname = f'sga_2025_data_{region}_{raslice}_{group_name}.sha256sum'
    cpath = dirpath / cname

    if cpath.exists():
        return 'skipped'

    files = sorted(f for f in dirpath.iterdir() if f.is_file() and f.name != cname)
    lines = [f'{sha256_file(f)}  {f.name}' for f in files]
    cpath.write_text('\n'.join(lines) + '\n')
    return 'done'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate SGA2025 sha256 checksums.')
    parser.add_argument('topdir', help='Region directory (e.g. /path/to/dr11-south)')
    parser.add_argument('--logfile', default='checksums.log',
                        help='Output log file (default: checksums.log)')
    args = parser.parse_args()

    topdir = Path(args.topdir)
    region = topdir.name  # e.g. 'dr11-south'

    # Logging on rank 0 only
    log = logging.getLogger('checksums')
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(message)s')
    if _rank == 0:
        log.addHandler(logging.StreamHandler(sys.stdout))
        log.addHandler(logging.FileHandler(args.logfile, mode='w'))
        for h in log.handlers:
            h.setFormatter(fmt)

    # Fast glob — validation has already passed so no need to filter with is_dir()
    if _rank == 0:
        all_dirs = sorted(glob(str(topdir / '???/*')))
        log.info(f'Region:    {region}')
        log.info(f'Topdir:    {topdir}')
        log.info(f'Groups:    {len(all_dirs)}')
        log.info(f'Log file:  {args.logfile}')
        if _use_mpi:
            log.info(f'MPI ranks: {_size}')
        log.info('')
    else:
        all_dirs = None

    if _use_mpi:
        all_dirs = _comm.bcast(all_dirs, root=0)

    my_dirs = all_dirs[_rank::_size]

    n_done = n_skipped = n_error = 0
    for i, d in enumerate(my_dirs):
        if _rank == 0 and (i + 1) % 5000 == 0:
            log.info(f'  {i+1}/{len(my_dirs)} ... done={n_done}, skipped={n_skipped}')

        dirpath = Path(d)
        raslice    = dirpath.parent.name  # e.g. '000'
        group_name = dirpath.name         # e.g. '00000m0649'

        try:
            status = process_group(dirpath, region, raslice, group_name)
            if status == 'skipped':
                n_skipped += 1
            else:
                n_done += 1
        except Exception as e:
            # Don't let one bad group abort the whole run
            print(f'[rank {_rank:03d}] ERROR {group_name}: {e}', flush=True)
            n_error += 1

    # Gather totals on rank 0
    if _use_mpi:
        all_totals = _comm.gather((n_done, n_skipped, n_error), root=0)
        if _rank == 0:
            n_done    = sum(t[0] for t in all_totals)
            n_skipped = sum(t[1] for t in all_totals)
            n_error   = sum(t[2] for t in all_totals)

    if _rank == 0:
        log.info('')
        log.info('=' * 50)
        log.info('CHECKSUM SUMMARY')
        log.info('=' * 50)
        log.info(f'  Written:  {n_done:6d}')
        log.info(f'  Skipped:  {n_skipped:6d}')
        log.info(f'  Errors:   {n_error:6d}')
        log.info('=' * 50)
        sys.exit(1 if n_error else 0)


if __name__ == '__main__':
    main()
