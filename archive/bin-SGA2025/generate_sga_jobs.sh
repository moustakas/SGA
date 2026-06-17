#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: generate_sga_jobs.sh -t TEMPLATE [--min MIN] [--max MAX] -n N
       [--spacing linear|log10] [--submit] [--stats]
       [--diam-file FILE] [--power P] [--count-file FILE]
       [--by-index] [--nsample INT]
       [--galaxylist-file FILE]

Split a diameter range into N subranges and generate N .slurm scripts.

Output directories:
  * Without --submit: writes to preview/ (overwrites existing files)
  * With --submit: writes to jobs/ with unique counter, then submits via sbatch

Default (diameter mode):
  * If --diam-file is provided, bins are formed so each job has ~equal sum(d^P),
    with P = --power (default 1.0). With --diam-file:
      - If MIN and/or MAX are omitted, infer from the file (positives only).
      - If MIN and/or MAX are provided, those exact values are used as the
        outer bin edges; weights still come from entries within [MIN,MAX].
  * If --diam-file is not provided, you must supply --min and --max and pick
    --spacing (linear|log10; default linear).

Index mode (when --by-index is given):
  * Jobs are generated using --first/--last (half-open indices).
  * nsample is determined in this order:
      1) --count-file line count (if provided),
      2) else --diam-file line count (if provided),
      3) else require --nsample.
  * The sample is split as evenly as possible across N jobs.

Galaxylist mode (when --galaxylist-file is given):
  * File format: name,diameter (one per line, comma-separated).
  * Jobs are generated using --galaxylist with a comma-separated list of names.
  * The galaxy list is split so each job has ~equal sum(d^P), with P = --power.
  * When using galaxylist mode, any __MINDIAM__/__MAXDIAM__ and __FIRST__/__LAST__
    lines in the TEMPLATE are dropped.

Counts:
  - If --count-file is provided, the script prints [NN objects] for each bin,
    counting lines in that file whose diameter lies inside the bin (diameter mode),
    or reporting the index-span count in index mode.
  - If --count-file is omitted but --diam-file is present, the --diam-file is used
    for counting. If neither is provided, counts are omitted.

Template injection:
  * Diameter mode:
      - If TEMPLATE contains __MINDIAM__ and __MAXDIAM__, they are replaced.
      - Otherwise, flags are appended to the last line containing 'SGA2025-mpi'
        or 'SGA2025-mpi.sh'.
  * Index mode:
      - If TEMPLATE contains __FIRST__ and __LAST__, they are replaced.
      - Otherwise, flags --first/--last are appended to the last SGA2025-mpi line.

Examples:

Galaxylist mode (preview):
./generate_sga_jobs.sh \
  -t coadds.template.slurm -n 24 \
  --galaxylist-file galaxylist-coadds-dr11-south.txt

Galaxylist mode (submit):
./generate_sga_jobs.sh \
  -t coadds.template.slurm -n 24 \
  --galaxylist-file galaxylist-coadds-dr11-south.txt --submit

Weighted bins (infer bounds) + counts from same file:
./generate_sga_jobs.sh \
  -t v0.20-coadds.template.slurm -n 4 \
  --diam-file remaining-dr9-north.txt --power 2.0

Linear spacing with explicit bounds:
./generate_sga_jobs.sh \
  -t v0.20-coadds.template.slurm -n 4 \
  --min 1.0 --max 10.0 --spacing linear

Index mode, use diam-file's length as nsample:
./generate_sga_jobs.sh \
  -t v0.20-coadds.template.slurm -n 8 \
  --diam-file remaining.txt --by-index
EOF
}

# ------------------ Parse args ------------------
TEMPLATE=""; MIN=""; MAX=""; NJOBS=""
PREFIX=""; SUBMIT=0; SPACING="linear"; ADD_STATS=0
DIAM_FILE=""; POWER="1.0"; COUNT_FILE=""
BY_INDEX=0; NSAMPLE=""
GALAXYLIST_FILE=""

while (( "$#" )); do
  case "$1" in
    -t|--template) TEMPLATE=$2; shift 2;;
    --min)         MIN=$2; shift 2;;
    --max)         MAX=$2; shift 2;;
    -n|--njobs)    NJOBS=$2; shift 2;;
    --prefix)      PREFIX=$2; shift 2;;
    --spacing)     SPACING=$2; shift 2;;
    --submit)      SUBMIT=1; shift;;
    --stats)       ADD_STATS=1; shift;;
    --diam-file)   DIAM_FILE=$2; shift 2;;
    --power)       POWER=$2; shift 2;;
    --count-file)  COUNT_FILE=$2; shift 2;;
    --by-index)    BY_INDEX=1; shift;;
    --nsample)     NSAMPLE=$2; shift 2;;
    --galaxylist-file) GALAXYLIST_FILE=$2; shift 2;;
    -h|--help)     usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

# ------------------ Required args ------------------
if [[ -n "$GALAXYLIST_FILE" ]]; then
  [[ -n "$TEMPLATE" && -n "$NJOBS" ]] || { usage; exit 2; }
elif (( BY_INDEX )); then
  [[ -n "$TEMPLATE" && -n "$NJOBS" ]] || { usage; exit 2; }
else
  if [[ -n "$DIAM_FILE" ]]; then
    [[ -n "$TEMPLATE" && -n "$NJOBS" ]] || { usage; exit 2; }
  else
    [[ -n "$TEMPLATE" && -n "$MIN" && -n "$MAX" && -n "$NJOBS" ]] || { usage; exit 2; }
  fi
fi
[[ -r "$TEMPLATE" ]] || { echo "Template not readable: $TEMPLATE" >&2; exit 2; }

# Extract STAGE, REGION, and constraint from template for filename generation
extract_template_info() {
  local tpl="$1"
  # Extract STAGE= value (handles STAGE=xxx or STAGE="xxx")
  T_STAGE=$(awk -F= '/^STAGE=/ {gsub(/[" ]/, "", $2); print $2; exit}' "$tpl")
  # Extract REGION= value
  T_REGION=$(awk -F= '/^REGION=/ {gsub(/[" ]/, "", $2); print $2; exit}' "$tpl")
  # Extract --constraint= value from #SBATCH line
  T_CONSTRAINT=$(grep '^#SBATCH.*--constraint=' "$tpl" | sed -n 's/.*--constraint=\([^ "]*\).*/\1/p' | head -1)

  # Fallbacks if not found
  : "${T_STAGE:=stage}"
  : "${T_REGION:=region}"
  : "${T_CONSTRAINT:=run}"
}

# Find next available counter for output files
# Usage: find_next_counter BASE_PATTERN OUTDIR
# BASE_PATTERN should contain %d placeholder for counter
find_next_counter() {
  local pattern="$1" outdir="$2"
  local counter=0
  while true; do
    local test_file="${outdir}/$(printf "$pattern" "$counter")"
    if [[ ! -e "$test_file" ]]; then
      echo "$counter"
      return
    fi
    (( counter++ ))
  done
}

# Generate output filename
# Usage: make_output_filename JOB_INDEX NJOBS SUBMIT_MODE
make_output_filename() {
  local job_idx="$1" njobs="$2" submit_mode="$3"

  # Determine padding width based on njobs
  local width=${#njobs}
  (( width < 3 )) && width=3

  local job_str=$(printf "%0${width}d" "$job_idx")
  local njobs_str=$(printf "%0${width}d" "$njobs")

  if (( submit_mode )); then
    # Submit mode: use jobs/ subdir with counter for uniqueness
    mkdir -p "jobs"

    # Find next available counter (only check for job 0 to keep set consistent)
    if [[ -z "${_BATCH_COUNTER:-}" ]]; then
      _BATCH_COUNTER=$(find_next_counter "${T_STAGE}.${T_REGION}.${T_CONSTRAINT}.${job_str}_of_${njobs_str}.%d.slurm" "jobs")
    fi

    echo "jobs/${T_STAGE}.${T_REGION}.${T_CONSTRAINT}.${job_str}_of_${njobs_str}.${_BATCH_COUNTER}.slurm"
  else
    # Preview mode: use preview/ subdir, overwrite existing (no counter)
    mkdir -p "preview"

    echo "preview/${T_STAGE}.${T_REGION}.${T_CONSTRAINT}.${job_str}_of_${njobs_str}.slurm"
  fi
}

extract_template_info "$TEMPLATE"

contains_tokens=0
contains_tokens_idx=0
contains_tokens_gal=0
if grep -q '__MINDIAM__' "$TEMPLATE" && grep -q '__MAXDIAM__' "$TEMPLATE"; then
  contains_tokens=1
fi
if grep -q '__FIRST__' "$TEMPLATE" && grep -q '__LAST__' "$TEMPLATE"; then
  contains_tokens_idx=1
fi
if grep -q '__GALAXYLIST__' "$TEMPLATE"; then
  contains_tokens_gal=1
fi

# ------------------ Helpers ------------------
fmt5() { awk -v x="$1" 'BEGIN{ printf "%.5f", x }'; }

append_stats_footer() {
  local of="$1"
  cat >>"$of" <<'STATS'

echo ""
echo "===== Job summary (sacct) ====="
if command -v sacct >/dev/null 2>&1; then
  sleep 2
  sacct -j "$SLURM_JOB_ID" -X -n -o JobID,JobName%30,State,Elapsed,TotalCPU,MaxRSS,AveRSS,MaxVMSize,ReqMem,NodeList -P
else
  echo "sacct not available on this system."
fi
STATS
}

make_one_diam() {
  local mind="$2" maxd="$3" ofile="$4"

  if (( contains_tokens == 1 )); then
    # Fill __MINDIAM__/__MAXDIAM__; drop any line that still has __FIRST__/__LAST__
    awk -v mi="$(fmt5 "$mind")" -v ma="$(fmt5 "$maxd")" '
      {
        gsub(/__MINDIAM__/, mi)
        gsub(/__MAXDIAM__/, ma)
        if ($0 ~ /__GALAXYLIST__/ || $0 ~ /--galaxylist=/ ) next
        if ($0 ~ /__(FIRST|LAST)__/ ) next
        print
      }
    ' "$TEMPLATE" > "$ofile"
  else
    cp "$TEMPLATE" "$ofile"

    # Drop any galaxylist placeholder line in diameter mode

    awk '{ if ($0 ~ /__GALAXYLIST__/ || $0 ~ /--galaxylist=/ ) next; print }' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    local ln
    ln=$(awk '/SGA2025-mpi(\.sh)?($|[[:space:]])/ {last=NR} END{ if(last) print last; else print 0 }' "$ofile")
    if [[ "$ln" -eq 0 ]]; then
      # No call site found; still replace tokens if present and drop __FIRST__/__LAST__ lines
      awk -v mi="$(fmt5 "$mind")" -v ma="$(fmt5 "$maxd")" '
        {
          gsub(/__MINDIAM__/, mi)
          gsub(/__MAXDIAM__/, ma)
          if ($0 ~ /__GALAXYLIST__/ || $0 ~ /--galaxylist=/ ) next
          if ($0 ~ /__(FIRST|LAST)__/ ) next
          print
        }
      ' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    else
      sed -i "${ln}s@\$@ --mindiam=$(fmt5 "$mind") --maxdiam=$(fmt5 "$maxd")@" "$ofile"
    fi
  fi

  (( ADD_STATS )) && append_stats_footer "$ofile"
  chmod +x "$ofile"
}

make_one_index() {
  local first="$2" last="$3" ofile="$4"

  if (( contains_tokens_idx == 1 )); then
    # Fill __FIRST__/__LAST__; drop any line with __MINDIAM__/__MAXDIAM__ or __GALAXYLIST__
    awk -v fi="$first" -v la="$last" '
      {
        gsub(/__FIRST__/, fi)
        gsub(/__LAST__/,  la)
        if ($0 ~ /__(MINDIAM|MAXDIAM)__/ ) next
        if ($0 ~ /__GALAXYLIST__/ || $0 ~ /--galaxylist=/ ) next
        print
      }
    ' "$TEMPLATE" > "$ofile"
  else
    cp "$TEMPLATE" "$ofile"
    # Drop any placeholder lines for other modes
    awk '
      {
        if ($0 ~ /__GALAXYLIST__/ || $0 ~ /--galaxylist=/ ) next
        if ($0 ~ /__(MINDIAM|MAXDIAM)__/ ) next
        print
      }
    ' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    local ln
    ln=$(awk '/SGA2025-mpi(\.sh)?($|[[:space:]])/ {last=NR} END{ if(last) print last; else print 0 }' "$ofile")
    if [[ "$ln" -eq 0 ]]; then
      # No call site found; still replace tokens if present
      awk -v fi="$first" -v la="$last" '
        {
          gsub(/__FIRST__/, fi)
          gsub(/__LAST__/,  la)
          print
        }
      ' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    else
      sed -i "${ln}s@\$@ --first=$first --last=$last@" "$ofile"
    fi
  fi

  (( ADD_STATS )) && append_stats_footer "$ofile"
  chmod +x "$ofile"
}

make_one_galaxylist() {
  local gcsv="$2" ofile="$3"

  if (( contains_tokens_gal == 1 )); then
    # Fill __GALAXYLIST__; drop any line that still has __MINDIAM__/__MAXDIAM__ or __FIRST__/__LAST__
    awk -v gl="$gcsv" '
      {
        gsub(/__GALAXYLIST__/, gl)
        if ($0 ~ /__(MINDIAM|MAXDIAM|FIRST|LAST)__/ ) next
        print
      }
    ' "$TEMPLATE" > "$ofile"
  else
    cp "$TEMPLATE" "$ofile"
    local ln
    ln=$(awk '/SGA2025-mpi(\.sh)?($|[[:space:]])/ {last=NR} END{ if(last) print last; else print 0 }' "$ofile")
    if [[ "$ln" -eq 0 ]]; then
      # No call site found; still drop any placeholder lines
      awk -v gl="$gcsv" '
        {
          if ($0 ~ /__(MINDIAM|MAXDIAM|FIRST|LAST)__/ ) next
          print
        }
      ' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    else
      # Drop any leftover placeholder lines (in case template includes them elsewhere)
      awk '
        { if ($0 ~ /__(MINDIAM|MAXDIAM|FIRST|LAST)__/ ) next; print }
      ' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
      # Re-find callsite line number after dropping placeholder lines
      ln=$(awk '/SGA2025-mpi(\.sh)?($|[[:space:]])/ {last=NR} END{ if(last) print last; else print 0 }' "$ofile")
      if [[ "$ln" -eq 0 ]]; then
        echo "ERROR: could not find SGA2025-mpi callsite in $ofile" >&2
        exit 2
      fi
      sed -i "${ln}s@\$@ --galaxylist=$gcsv@" "$ofile"
    fi
  fi

  (( ADD_STATS )) && append_stats_footer "$ofile"
  chmod +x "$ofile"
}


# Helper: count how many items in COUNT_SRC fall within [lo, hi]
count_in_range() {
  local lo="$1" hi="$2" src="$3" is_last="${4:-0}"
  awk -v lo="$lo" -v hi="$hi" -v is_last="$is_last" '
    { d=$1+0; if (d>=lo && (is_last ? d<=hi : d<hi)) c++ }
    END{ print (c+0) }
  ' "$src"
}

# Compute sum of d^p for diameters in range [lo, hi) or [lo, hi] if last bin
weight_in_range() {
  local lo="$1" hi="$2" src="$3" p="$4" is_last="${5:-0}"
  awk -v lo="$lo" -v hi="$hi" -v p="$p" -v is_last="$is_last" '
    function pw(x,p){ return (p==1 ? x : (p==2 ? x*x : exp(p*log(x)))) }
    { d=$1+0; if (d>=lo && (is_last ? d<=hi : d<hi) && d>0) w += pw(d,p) }
    END{ printf "%.2f", w+0 }
  ' "$src"
}

# Compute total weight for a file
total_weight_in_file() {
  local src="$1" p="$2"
  awk -v p="$p" '
    function pw(x,p){ return (p==2 ? x*x : exp(p*log(x))) }
    { d=$1+0; if (d>0) w += pw(d,p) }
    END{ printf "%.2f", w+0 }
  ' "$src"
}

# Decide which file to use for counts (diameter mode), if any
if [[ -z "$COUNT_FILE" && -n "$DIAM_FILE" ]]; then
  COUNT_FILE="$DIAM_FILE"
fi
if [[ -n "$COUNT_FILE" && ! -r "$COUNT_FILE" ]]; then
  echo "WARN: --count-file provided but not readable: $COUNT_FILE; counts will be omitted." >&2
  COUNT_FILE=""
fi

# ------------------ Index mode ------------------
if (( BY_INDEX )); then
  # Determine nsample
  local_count_src=""
  if [[ -n "$COUNT_FILE" && -r "$COUNT_FILE" ]]; then
    local_count_src="$COUNT_FILE"
  elif [[ -n "$DIAM_FILE" && -r "$DIAM_FILE" ]]; then
    local_count_src="$DIAM_FILE"
  fi

  if [[ -n "$local_count_src" ]]; then
    NSAMPLE=$(wc -l < "$local_count_src")
  fi

  if [[ -z "$NSAMPLE" ]]; then
    echo "ERROR: --by-index requires either --count-file, --diam-file, or --nsample." >&2
    exit 2
  fi

  awk -v n="$NJOBS" -v m="$NSAMPLE" 'BEGIN{
    if (n != int(n) || n < 1) { print "NJOBS must be integer >= 1" > "/dev/stderr"; exit 2 }
    if (m != int(m) || m < 1) { print "NSAMPLE must be integer >= 1" > "/dev/stderr"; exit 2 }
  }' || exit 2

  echo "Generating $NJOBS jobs by index over nsample=$NSAMPLE"

  # Print total weight if we have a source file
  if [[ -n "$local_count_src" ]]; then
    total_w=$(total_weight_in_file "$local_count_src" "$POWER")
    target_w=$(awk -v t="$total_w" -v n="$NJOBS" 'BEGIN{ printf "%.2f", t/n }')
    echo "  Total weight: $total_w, target per job: $target_w"
  fi

  declare -a outfiles
  # Even split of [0, NSAMPLE)
  q=$(( NSAMPLE / NJOBS ))
  r=$(( NSAMPLE % NJOBS ))
  start=0
  for (( i=0; i<NJOBS; i++ )); do
    len=$q; if (( i < r )); then len=$(( q + 1 )); fi
    first=$start
    last=$(( start + len ))
    start=$last

    ofile=$(make_output_filename "$i" "$NJOBS" "$SUBMIT")
    make_one_index "$i" "$first" "$last" "$ofile"
    outfiles+=( "$ofile" )

    # Compute weight for this index range if we have a source file
    if [[ -n "$local_count_src" ]]; then
      wbin=$(awk -v first="$first" -v last="$last" -v p="$POWER" '
        function pw(x,p){ return (p==2 ? x*x : exp(p*log(x))) }
        NR > first && NR <= last { d=$1+0; if(d>0) w += pw(d,p) }
        END{ printf "%.2f", w+0 }
      ' "$local_count_src")
      printf "  [%2d/%2d] %-50s  [%3d objects, weight=%7s]\n" \
             "$((i+1))" "$NJOBS" "$ofile" "$len" "$wbin"
    else
      printf "  [%2d/%2d] %-50s  --first=%d  --last=%d     [%d objects]\n" \
             "$((i+1))" "$NJOBS" "$ofile" "$first" "$last" "$len"
    fi
  done

  if (( SUBMIT )); then
    echo "Submitting ${#outfiles[@]} jobs with sbatch..."
    for f in "${outfiles[@]}"; do
      sbatch "$f"
    done
  fi
  exit 0
fi

# ------------------ Galaxylist mode ------------------
if [[ -n "$GALAXYLIST_FILE" ]]; then
  [[ -r "$GALAXYLIST_FILE" ]] || { echo "galaxylist-file not readable: $GALAXYLIST_FILE" >&2; exit 2; }

  # Read names and diameters from "name,diameter" format (skip blank lines)
  mapfile -t GALS < <(awk -F, 'NF && $1!="" {print $1}' "$GALAXYLIST_FILE")
  mapfile -t DIAMS < <(awk -F, 'NF && $1!="" {print $2+0}' "$GALAXYLIST_FILE")
  NGALS=${#GALS[@]}
  if (( NGALS == 0 )); then
    echo "ERROR: galaxylist-file has no usable rows: $GALAXYLIST_FILE" >&2
    exit 2
  fi

  awk -v n="$NJOBS" -v m="$NGALS" 'BEGIN{
    if (n != int(n) || n < 1) { print "NJOBS must be integer >= 1" > "/dev/stderr"; exit 2 }
    if (m != int(m) || m < 1) { print "NGALS must be integer >= 1" > "/dev/stderr"; exit 2 }
  }' || exit 2

  # Compute weighted bin assignments (same algorithm as diameter mode)
  # Output: one line per job with space-separated indices
  mapfile -t JOB_INDICES < <(
    for (( k=0; k<NGALS; k++ )); do
      echo "${GALS[k]},${DIAMS[k]}"
    done | awk -F, -v p="$POWER" -v N="$NJOBS" '
      function pw(x,p){ return (p==2 ? x*x : exp(p*log(x))) }
      {
        name[NR] = $1
        diam[NR] = $2 + 0
        if (diam[NR] <= 0) diam[NR] = 0.001
      }
      END {
        n = NR
        if (n == 0) exit

        # Compute total weight
        W = 0
        for (i=1; i<=n; i++) {
          w[i] = pw(diam[i], p)
          W += w[i]
        }
        if (W == 0) W = 1

        target = W / N
        cumul = 0
        job = 0
        job_indices[job] = ""

        for (i=1; i<=n; i++) {
          cumul += w[i]
          if (job_indices[job] == "") {
            job_indices[job] = (i-1)
          } else {
            job_indices[job] = job_indices[job] " " (i-1)
          }
          # Move to next job if we have exceeded target and not on last job
          if (cumul >= (job+1) * target && job < N-1) {
            job++
            job_indices[job] = ""
          }
        }

        # Output one line per job (space-separated 0-based indices)
        for (j=0; j<N; j++) {
          if (job_indices[j] == "") {
            print ""
          } else {
            print job_indices[j]
          }
        }
      }
    '
  )

  echo "Generating $NJOBS jobs from galaxy list '$GALAXYLIST_FILE' (ngals=$NGALS, power=$POWER)"

  # Compute total weight in one awk call
  total_weight=$(printf '%s\n' "${DIAMS[@]}" | awk -v p="$POWER" '
    function pw(x,p){ return (p==1 ? x : (p==2 ? x*x : exp(p*log(x)))) }
    { d=$1+0; if(d>0) w += pw(d,p) }
    END{ printf "%.2f", w+0 }
  ')
  target_weight=$(awk -v t="$total_weight" -v n="$NJOBS" 'BEGIN{ printf "%.2f", t/n }')
  echo "  Total weight: $total_weight, target per job: $target_weight"

  # Precompute all weights in one awk call
  mapfile -t WEIGHTS < <(printf '%s\n' "${DIAMS[@]}" | awk -v p="$POWER" '
    function pw(x,p){ return (p==1 ? x : (p==2 ? x*x : exp(p*log(x)))) }
    { d=$1+0; printf "%.6f\n", (d>0 ? pw(d,p) : 0) }
  ')

  declare -a outfiles
  for (( i=0; i<NJOBS; i++ )); do
    indices_str="${JOB_INDICES[i]}"
    if [[ -z "$indices_str" ]]; then
      gcsv=""
      len=0
      job_weight="0.00"
    else
      # Build comma-separated list of names and sum weights using single awk
      read -ra idx_arr <<< "$indices_str"
      len=${#idx_arr[@]}
      gcsv=""
      weight_sum=""
      for idx in "${idx_arr[@]}"; do
        if [[ -z "$gcsv" ]]; then
          gcsv="${GALS[idx]}"
          weight_sum="${WEIGHTS[idx]}"
        else
          gcsv="${gcsv},${GALS[idx]}"
          weight_sum="${weight_sum} ${WEIGHTS[idx]}"
        fi
      done
      job_weight=$(echo "$weight_sum" | awk '{ for(i=1;i<=NF;i++) s+=$i } END{ printf "%.2f", s }')
    fi

    ofile=$(make_output_filename "$i" "$NJOBS" "$SUBMIT")
    make_one_galaxylist "$i" "$gcsv" "$ofile"
    outfiles+=( "$ofile" )

    printf "  [%2d/%2d] %-50s  [%3d galaxies, weight=%7s]\n" \
           "$((i+1))" "$NJOBS" "$ofile" "$len" "$job_weight"
  done

  if (( SUBMIT )); then
    echo "Submitting ${#outfiles[@]} jobs with sbatch..."
    for f in "${outfiles[@]}"; do
      sbatch "$f"
    done
  fi
  exit 0
fi

# ------------------ Diameter mode (existing behavior) ------------------

# If diam-file: infer any missing min/max from the file (positives only)
if [[ -n "$DIAM_FILE" ]]; then
  [[ -r "$DIAM_FILE" ]] || { echo "diam-file not readable: $DIAM_FILE" >&2; exit 2; }
  if [[ -z "$MIN" || -z "$MAX" ]]; then
    read -r fmin fmax < <(
      awk '{d=$1+0; if(d>0){ if(!seen++){min=d; max=d}else{ if(d<min)min=d; if(d>max)max=d }}} END{ if(seen==0) exit 3; printf "%.17g %.17g\n", min, max }' \
      "$DIAM_FILE"
    ) || { echo "ERROR: could not infer min/max from $DIAM_FILE" >&2; exit 2; }
    [[ -n "$MIN" ]] || MIN="$fmin"
    [[ -n "$MAX" ]] || MAX="$fmax"
  fi
fi

# Sanity checks
awk -v min="$MIN" -v max="$MAX" -v n="$NJOBS" '
BEGIN{
  if (n != int(n) || n < 1) { print "NJOBS must be integer >= 1" > "/dev/stderr"; exit 2 }
  if (!(min < max)) { printf "Require --min < --max (got %s vs %s)\n", min, max > "/dev/stderr"; exit 2 }
}' || exit 2

case "$SPACING" in
  linear) ;;
  log10)
    awk -v min="$MIN" 'BEGIN{ if (min <= 0) { print "For --spacing log10, --min must be > 0" > "/dev/stderr"; exit 2 } }' || exit 2
    ;;
  *) echo "Unknown --spacing: $SPACING (use 'linear' or 'log10')" >&2; exit 2;;
esac

declare -a outfiles


if [[ -n "$DIAM_FILE" ]]; then
  echo "Generating up to $NJOBS jobs from diameter list '$DIAM_FILE' with p=$POWER over [$(fmt5 "$MIN"), $(fmt5 "$MAX")]"

  mapfile -t EDGES < <(
    awk -v min="$MIN" -v max="$MAX" '{ d=$1+0; if(d>0 && d>=min && d<=max) print d }' "$DIAM_FILE" \
    | sort -g \
    | awk -v p="$POWER" -v N="$NJOBS" -v min="$MIN" -v max="$MAX" '
        function pw(x,p){ return (p==1 ? x : (p==2 ? x*x : exp(p*log(x)))) }
        { x[++n]=$1 }
        END{
          if(n==0){ print "ERROR: no diameters fall within the requested [min,max] range." > "/dev/stderr"; exit 3 }
          W=0; for(i=1;i<=n;i++){ w[i]=pw(x[i],p); W+=w[i] }
          if(W==0){ print "ERROR: total weight is zero." > "/dev/stderr"; exit 3 }

          target=W/N
          sc=0; c=0; k=1
          for(i=1;i<n && k<N; i++){
            c += w[i]
            if(c >= k*target){ sc++; sidx[sc]=i; k++ }
          }

          # Build initial bins with their index ranges
          lo=1
          nb=0
          for(b=1;b<=N;b++){
            if(b<=sc){ hi=sidx[b] } else { hi=n }
            if(hi>=lo){
              nb++
              bin_lo[nb]=lo
              bin_hi[nb]=hi
            }
            lo = hi+1
          }

          # Compute ranges for all bins
          for(b=1;b<=nb;b++){
            lo_idx = bin_lo[b]; hi_idx = bin_hi[b]
            if (b==1) bin_min[b]=min+0; else bin_min[b]=x[lo_idx]+0
            if (b==nb) bin_max[b]=max+0; else bin_max[b]=x[hi_idx]+0
          }

          # Output bins, merging/skipping as needed
          # Track the last output max to avoid overlapping ranges
          last_out_max = -1
          out_n = 0

          for(b=1;b<=nb;b++){
            mn = bin_min[b]
            mx = bin_max[b]
            hi_idx = bin_hi[b]

            # Skip if this bin starts before or at the last output max
            # (meaning it would overlap or duplicate)
            if (out_n > 0 && mn <= last_out_max) {
              # Extend the previous output bin instead
              # We need to update last_out_max if this bin extends further
              if (mx > last_out_max) {
                # This bin extends beyond - we cant easily update previous output
                # So just update mn to start after last_out_max
                mn = last_out_max
              } else {
                # Entire bin is within previous range, skip
                continue
              }
            }

            # Ensure min < max
            if (mn >= mx) {
              # Find next distinct value
              found = 0
              for (j = hi_idx + 1; j <= n; j++) {
                if (x[j]+0 > mn) {
                  mx = x[j]+0
                  found = 1
                  break
                }
              }
              if (!found) {
                if (mn >= (max+0) - 0.00001) {
                  mx = (max+0) * 1.00001
                } else {
                  mx = max+0
                }
              }
            }

            # Final check: skip if still invalid or would overlap
            if (mn >= mx) continue
            if (out_n > 0 && mn < last_out_max) {
              mn = last_out_max
              if (mn >= mx) continue
            }

            # For the last bin (b == nb), add small margin to max
            # to ensure objects exactly at max are included with semi-open [min, max)
            if (b == nb && mx <= (max+0) + 0.00001) {
              mx = mx * 1.00001
            }

            out_n++
            printf("%.5f %.5f\n", mn, mx)
            last_out_max = mx
          }
        }'
  ) || { printf '%s\n' "${EDGES[@]}" >&2; exit 4; }

  if (( ${#EDGES[@]} == 0 )); then
    echo "ERROR: no valid bins produced (all bins empty or have min>=max); aborting." >&2
    exit 4
  fi

  # Update NJOBS to actual number of bins produced
  actual_njobs=${#EDGES[@]}
  if (( actual_njobs != NJOBS )); then
    echo "  Note: reduced to $actual_njobs jobs (merged bins with identical ranges)"
  fi

  # Print total weight and target
  total_w=$(total_weight_in_file "$DIAM_FILE" "$POWER")
  target_w=$(awk -v t="$total_w" -v n="$actual_njobs" 'BEGIN{ printf "%.2f", t/n }')
  echo "  Total weight: $total_w, target per job: $target_w"

  for (( i=0; i<${#EDGES[@]}; i++ )); do
    read -r mind maxd <<<"${EDGES[i]}"
    ofile=$(make_output_filename "$i" "$actual_njobs" "$SUBMIT")
    make_one_diam "$i" "$mind" "$maxd" "$ofile"
    outfiles+=( "$ofile" )
    if [[ -n "$COUNT_FILE" ]]; then
      is_last=0; (( i == ${#EDGES[@]} - 1 )) && is_last=1
      nbin=$(count_in_range "$mind" "$maxd" "$COUNT_FILE" "$is_last")
      wbin=$(weight_in_range "$mind" "$maxd" "$COUNT_FILE" "$POWER" "$is_last")
      printf "  [%2d/%2d] %-50s  [%3d objects, weight=%7s]\n" \
        "$((i+1))" "$actual_njobs" "$ofile" "$nbin" "$wbin"
    else
      printf "  [%2d/%2d] %-50s  --mindiam=%.5f  --maxdiam=%.5f\n" \
        "$((i+1))" "$actual_njobs" "$ofile" "$mind" "$maxd"
    fi
  done

elif [[ "$SPACING" == "linear" ]]; then
  step=$(awk -v a="$MIN" -v b="$MAX" -v n="$NJOBS" 'BEGIN{ printf "%.17g", (b-a)/n }')
  printf "Generating %d jobs (linear) over [%.5f, %.5f] step %.5f\n" \
         "$NJOBS" "$(fmt5 "$MIN")" "$(fmt5 "$MAX")" "$(fmt5 "$step")"

  # Print total weight and target if count file available
  if [[ -n "$COUNT_FILE" ]]; then
    total_w=$(total_weight_in_file "$COUNT_FILE" "$POWER")
    target_w=$(awk -v t="$total_w" -v n="$NJOBS" 'BEGIN{ printf "%.2f", t/n }')
    echo "  Total weight: $total_w, target per job: $target_w"
  fi

  for (( i=0; i<NJOBS; i++ )); do
    mind=$(awk -v a="$MIN" -v s="$step" -v i="$i" 'BEGIN{ printf "%.17g", a + i*s }')
    if (( i == NJOBS-1 )); then
      maxd="$MAX"
    else
      maxd=$(awk -v a="$MIN" -v s="$step" -v i="$i" 'BEGIN{ printf "%.17g", a + (i+1)*s }')
    fi
    ofile=$(make_output_filename "$i" "$NJOBS" "$SUBMIT")
    make_one_diam "$i" "$mind" "$maxd" "$ofile"
    outfiles+=( "$ofile" )
    if [[ -n "$COUNT_FILE" ]]; then
      is_last=0; (( i == NJOBS - 1 )) && is_last=1
      nbin=$(count_in_range "$mind" "$maxd" "$COUNT_FILE" "$is_last")
      wbin=$(weight_in_range "$mind" "$maxd" "$COUNT_FILE" "$POWER" "$is_last")
      printf "  [%2d/%2d] %-50s  [%3d objects, weight=%7s]\n" \
             "$((i+1))" "$NJOBS" "$ofile" "$nbin" "$wbin"
    else
      printf "  [%2d/%2d] %-50s  --mindiam=%s  --maxdiam=%s\n" \
             "$((i+1))" "$NJOBS" "$ofile" "$(fmt5 "$mind")" "$(fmt5 "$maxd")"
    fi
  done

else
  ln10=$(awk 'BEGIN{ print log(10) }')
  log10() { awk -v x="$1" -v l10="$ln10" 'BEGIN{ printf "%.17g", log(x)/l10 }'; }
  pow10() { awk -v y="$1" -v l10="$ln10" 'BEGIN{ printf "%.17g", exp(y*l10) }'; }

  lo=$(log10 "$MIN"); hi=$(log10 "$MAX")
  dlog=$(awk -v a="$lo" -v b="$hi" -v n="$NJOBS" 'BEGIN{ printf "%.17g", (b-a)/n }')
  echo "Generating $NJOBS jobs (log10) over [$(fmt5 "$MIN"), $(fmt5 "$MAX")]"

  # Print total weight and target if count file available
  if [[ -n "$COUNT_FILE" ]]; then
    total_w=$(total_weight_in_file "$COUNT_FILE" "$POWER")
    target_w=$(awk -v t="$total_w" -v n="$NJOBS" 'BEGIN{ printf "%.2f", t/n }')
    echo "  Total weight: $total_w, target per job: $target_w"
  fi

  for (( i=0; i<NJOBS; i++ )); do
    l1=$(awk -v a="$lo" -v s="$dlog" -v i="$i" 'BEGIN{ printf "%.17g", a + i*s }')
    if (( i == NJOBS-1 )); then
      l2="$hi"
    else
      l2=$(awk -v a="$lo" -v s="$dlog" -v i="$i" 'BEGIN{ printf "%.17g", a + (i+1)*s }')
    fi
    mind=$(pow10 "$l1")
    maxd=$(pow10 "$l2")

    ofile=$(make_output_filename "$i" "$NJOBS" "$SUBMIT")
    make_one_diam "$i" "$mind" "$maxd" "$ofile"
    outfiles+=( "$ofile" )
    if [[ -n "$COUNT_FILE" ]]; then
      is_last=0; (( i == NJOBS - 1 )) && is_last=1
      nbin=$(count_in_range "$mind" "$maxd" "$COUNT_FILE" "$is_last")
      wbin=$(weight_in_range "$mind" "$maxd" "$COUNT_FILE" "$POWER" "$is_last")
      printf "  [%2d/%2d] %-50s  [%3d objects, weight=%7s]\n" \
             "$((i+1))" "$NJOBS" "$ofile" "$nbin" "$wbin"
    else
      printf "  [%2d/%2d] %-50s  --mindiam=%s  --maxdiam=%s\n" \
             "$((i+1))" "$NJOBS" "$ofile" "$(fmt5 "$mind")" "$(fmt5 "$maxd")"
    fi
  done
fi

# ------------------ Optional submission ------------------
if (( SUBMIT )); then
  echo "Submitting ${#outfiles[@]} jobs with sbatch..."
  for f in "${outfiles[@]}"; do
    sbatch "$f"
  done
fi
