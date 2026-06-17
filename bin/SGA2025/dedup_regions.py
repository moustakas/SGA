"""
Deduplicate CSV rows by OBJNAME.

Rules:
- If a name appears with dr11-south AND another region: keep one row, clear REGION to ''.
- If a name appears with two non-dr11-south regions: keep both, print a WARNING.
- Singletons: pass through unchanged.
"""

import sys
import csv
from collections import defaultdict

def main(infile, outfile):
    with open(infile, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Group rows by OBJNAME
    by_name = defaultdict(list)
    for row in rows:
        by_name[row['OBJNAME']].append(row)

    output_rows = []
    n_deduped = 0
    n_flagged = 0

    for name, group in by_name.items():
        if len(group) == 1:
            output_rows.append(group[0])
            continue

        regions = [r['REGION'] for r in group]
        has_dr11 = any(r == 'dr11-south' for r in regions)
        non_dr11 = [r for r in regions if r != 'dr11-south']

        if has_dr11 and len(non_dr11) >= 1:
            # Expected case: clear REGION and keep one row
            row = next(r for r in group if r['REGION'] != 'dr11-south')
            row = dict(row)
            row['REGION'] = ''
            output_rows.append(row)
            n_deduped += 1
        elif not has_dr11:
            # Two non-dr11-south regions — flag and keep both
            print(f"WARNING: {name!r} has multiple non-dr11-south regions: {regions}")
            output_rows.extend(group)
            n_flagged += 1
        else:
            # Multiple dr11-south rows (unexpected) — keep first, warn
            print(f"WARNING: {name!r} has multiple dr11-south rows, keeping first")
            output_rows.append(group[0])
            n_flagged += 1

    with open(outfile, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Done: {len(rows)} rows -> {len(output_rows)} rows "
          f"({n_deduped} deduplicated, {n_flagged} flagged)")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.csv output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
