#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: generate_sga_jobs.sh -t TEMPLATE [-o OUTDIR] [--min MIN] [--max MAX] -n N
       [--spacing linear|log10] [--prefix NAME] [--submit] [--stats]
       [--diam-file FILE] [--power P] [--count-file FILE]
       [--by-index] [--nsample INT]

Split a diameter range into N subranges and generate N .slurm scripts.

Default (diameter mode):
  * If --diam-file is provided, bins are formed so each job has ~equal sum(d^P),
    with P = --power (default 2.0). With --diam-file:
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

Weighted bins (infer bounds) + counts from same file:
./generate_sga_jobs.sh \
  -t v0.20-coadds.template.slurm -o jobs -n 4 \
  --diam-file remaining-dr9-north.txt --power 2.0

Linear spacing with explicit bounds:
./generate_sga_jobs.sh \
  -t v0.20-coadds.template.slurm -o jobs -n 4 \
  --min 1.0 --max 10.0 --spacing linear

Log10 spacing (explicit min only, infer max) + counts:
./generate_sga_jobs.sh \
  -t v0.20-coadds.template.slurm -o jobs -n 4 \
  --diam-file remaining-dr9-north.txt --min 1.0 --spacing log10

Index mode, use diam-file’s length as nsample:
./generate_sga_jobs.sh \
  -t v0.20-coadds.template.slurm -o jobs -n 8 \
  --diam-file remaining.txt --by-index

Index mode, explicit nsample (no files):
./generate_sga_jobs.sh \
  -t v0.20-coadds.template.slurm -o jobs -n 24 \
  --by-index --nsample 100000
EOF
}

# ------------------ Parse args ------------------
TEMPLATE=""; OUTDIR="."; MIN=""; MAX=""; NJOBS=""
PREFIX=""; SUBMIT=0; SPACING="linear"; ADD_STATS=0
DIAM_FILE=""; POWER="2.0"; COUNT_FILE=""
BY_INDEX=0; NSAMPLE=""

while (( "$#" )); do
  case "$1" in
    -t|--template) TEMPLATE=$2; shift 2;;
    -o|--outdir)   OUTDIR=$2; shift 2;;
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
    -h|--help)     usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

# ------------------ Required args ------------------
if (( BY_INDEX )); then
  [[ -n "$TEMPLATE" && -n "$OUTDIR" && -n "$NJOBS" ]] || { usage; exit 2; }
else
  if [[ -n "$DIAM_FILE" ]]; then
    [[ -n "$TEMPLATE" && -n "$OUTDIR" && -n "$NJOBS" ]] || { usage; exit 2; }
  else
    [[ -n "$TEMPLATE" && -n "$OUTDIR" && -n "$MIN" && -n "$MAX" && -n "$NJOBS" ]] || { usage; exit 2; }
  fi
fi
[[ -r "$TEMPLATE" ]] || { echo "Template not readable: $TEMPLATE" >&2; exit 2; }
mkdir -p "$OUTDIR"

# default prefix from template basename (strip .slurm and trailing .template)
if [[ -z "$PREFIX" ]]; then
  base=$(basename "$TEMPLATE"); base="${base%.slurm}"; PREFIX="${base%.template}"
fi

contains_tokens=0
contains_tokens_idx=0
if grep -q '__MINDIAM__' "$TEMPLATE" && grep -q '__MAXDIAM__' "$TEMPLATE"; then
  contains_tokens=1
fi
if grep -q '__FIRST__' "$TEMPLATE" && grep -q '__LAST__' "$TEMPLATE"; then
  contains_tokens_idx=1
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
  local idx="$1" mind="$2" maxd="$3" ofile="$4"

  if (( contains_tokens == 1 )); then
    # Fill __MINDIAM__/__MAXDIAM__; drop any line that still has __FIRST__/__LAST__
    awk -v mi="$(fmt5 "$mind")" -v ma="$(fmt5 "$maxd")" '
      {
        gsub(/__MINDIAM__/, mi)
        gsub(/__MAXDIAM__/, ma)
        if ($0 ~ /__GALAXYLIST__/ ) next
        if ($0 ~ /--galaxylist=/ ) next
        if ($0 ~ /__(FIRST|LAST)__/ ) next
        print
      }
    ' "$TEMPLATE" > "$ofile"
  else
    cp "$TEMPLATE" "$ofile"
    local ln
    ln=$(awk '/SGA2025-mpi(\.sh)?\b/ {last=NR} END{ if(last) print last; else print 0 }' "$ofile")
    if [[ "$ln" -eq 0 ]]; then
      # No call site found; still replace tokens if present and drop __FIRST__/__LAST__ lines
      awk -v mi="$(fmt5 "$mind")" -v ma="$(fmt5 "$maxd")" '
        {
          gsub(/__MINDIAM__/, mi)
          gsub(/__MAXDIAM__/, ma)
          if ($0 ~ /__GALAXYLIST__/ ) next
          if ($0 ~ /--galaxylist=/ ) next
          if ($0 ~ /__(FIRST|LAST)__/ ) next
          print
        }
      ' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    else
      sed -i "${ln}s@\$@ --mindiam=$(fmt5 "$mind") --maxdiam=$(fmt5 "$maxd")@" "$ofile"
      # Ensure galaxylist placeholder line is dropped in non-galaxylist modes.
      awk '{ if ($0 ~ /__GALAXYLIST__/ ) next; if ($0 ~ /--galaxylist=/ ) next; print }' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    fi
  fi

  (( ADD_STATS )) && append_stats_footer "$ofile"
  chmod +x "$ofile"
}

make_one_index() {
  local idx="$1" first="$2" last="$3" ofile="$4"

  if (( contains_tokens_idx == 1 )); then
    # Fill __FIRST__/__LAST__; drop any line that still has __MINDIAM__/__MAXDIAM__
    awk -v fi="$first" -v la="$last" '
      {
        gsub(/__FIRST__/, fi)
        gsub(/__LAST__/,  la)
        if ($0 ~ /__GALAXYLIST__/ ) next
        if ($0 ~ /--galaxylist=/ ) next
        if ($0 ~ /__(MINDIAM|MAXDIAM)__/ ) next
        print
      }
    ' "$TEMPLATE" > "$ofile"
  else
    cp "$TEMPLATE" "$ofile"
    local ln
    ln=$(awk '/SGA2025-mpi(\.sh)?\b/ {last=NR} END{ if(last) print last; else print 0 }' "$ofile")
    if [[ "$ln" -eq 0 ]]; then
      # No call site found; still replace tokens if present and drop __MINDIAM__/__MAXDIAM__ lines
      awk -v fi="$first" -v la="$last" '
        {
          gsub(/__FIRST__/, fi)
          gsub(/__LAST__/,  la)
          if ($0 ~ /__GALAXYLIST__/ ) next
          if ($0 ~ /--galaxylist=/ ) next
          if ($0 ~ /__(MINDIAM|MAXDIAM)__/ ) next
          print
        }
      ' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    else
      sed -i "${ln}s@\$@ --first=$first --last=$last@" "$ofile"
      # Ensure galaxylist placeholder line is dropped in non-galaxylist modes.
      awk '{ if ($0 ~ /__GALAXYLIST__/ ) next; if ($0 ~ /--galaxylist=/ ) next; print }' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    fi
  fi

  (( ADD_STATS )) && append_stats_footer "$ofile"
  chmod +x "$ofile"
}

make_one_galaxylist() {
  local idx="$1" glist="$2" ofile="$3"

  if (( contains_tokens_gal == 1 )); then
    # Fill __GALAXYLIST__; drop any line that still has __MINDIAM__/__MAXDIAM__ or __FIRST__/__LAST__
    awk -v gl="$glist" '
      {
        gsub(/__GALAXYLIST__/, gl)
        if ($0 ~ /__(MINDIAM|MAXDIAM)__/ ) next
        if ($0 ~ /__(FIRST|LAST)__/ ) next
        print
      }
    ' "$TEMPLATE" > "$ofile"
  else
    cp "$TEMPLATE" "$ofile"
    local ln
    ln=$(awk '/SGA2025-mpi(\.sh)?\b/ {last=NR} END{ if(last) print last; else print 0 }' "$ofile")
    if [[ "$ln" -eq 0 ]]; then
      # No call site found; still replace token if present and drop other mode lines
      awk -v gl="$glist" '
        {
          gsub(/__GALAXYLIST__/, gl)
          if ($0 ~ /__(MINDIAM|MAXDIAM)__/ ) next
          if ($0 ~ /__(FIRST|LAST)__/ ) next
          print
        }
      ' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    else
      # Append the flag to the SGA2025-mpi line.
      sed -i "${ln}s@\$@ --galaxylist=${glist}@" "$ofile"
      # Drop any leftover token lines from other modes.
      awk '{ if ($0 ~ /__(MINDIAM|MAXDIAM|FIRST|LAST)__/ ) next; print }' "$ofile" > "$ofile.tmp" && mv "$ofile.tmp" "$ofile"
    fi
  fi

  (( ADD_STATS )) && append_stats_footer "$ofile"
  chmod +x "$ofile"
}


# Helper: count how many items in COUNT_SRC fall within [lo, hi]
count_in_range() {
  local lo="$1" hi="$2" src="$3"
  awk -v lo="$lo" -v hi="$hi" '
    { d=$1+0; if (d>=lo && d<=hi) c++ }
    END{ print (c+0) }
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

    ofile="${OUTDIR}/${PREFIX}.i$(printf "%02d" $i)_of_$(printf "%02d" $NJOBS).slurm"
    make_one_index "$i" "$first" "$last" "$ofile"
    outfiles+=( "$ofile" )

    # Count is just len in index mode
    printf "  [%2d/%2d] %-40s  --first=%d  --last=%d     [%d objects]\n" \
           "$((i+1))" "$NJOBS" "$(basename "$ofile")" "$first" "$last" "$len"
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

# (…unchanged diameter splitting code below …)
# [The rest of your original diameter-mode implementation stays exactly the same.]
# -- I’ve inlined it here for completeness --

if [[ -n "$DIAM_FILE" ]]; then
  echo "Generating $NJOBS jobs from diameter list '$DIAM_FILE' with p=$POWER over [$(fmt5 "$MIN"), $(fmt5 "$MAX")]"

  mapfile -t EDGES < <(
    awk -v min="$MIN" -v max="$MAX" '{ d=$1+0; if(d>0 && d>=min && d<=max) print d }' "$DIAM_FILE" \
    | sort -g \
    | awk -v p="$POWER" -v N="$NJOBS" -v min="$MIN" -v max="$MAX" '
        function pw(x,p){ return (p==2 ? x*x : exp(p*log(x))) }
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

          lo=1
          for(b=1;b<=N;b++){
            if(b<=sc){ hi=sidx[b] } else { hi=n }
            if(hi<lo){
              printf("ERROR: empty bin would result at bin %d (reduce -n or adjust inputs)\n", b) > "/dev/stderr"
              exit 4
            }
            if (b==1) mn=min; else mn=x[lo]
            if (b==N) mx=max; else mx=x[hi]
            printf("%.5f %.5f\n", mn, mx)
            lo = hi+1
          }
        }'
  ) || { printf '%s\n' "${EDGES[@]}" >&2; exit 4; }

  if (( ${#EDGES[@]} != NJOBS )); then
    echo "ERROR: expected $NJOBS bins but produced ${#EDGES[@]} bins; aborting." >&2
    exit 4
  fi

  for (( i=0; i<${#EDGES[@]}; i++ )); do
    read -r mind maxd <<<"${EDGES[i]}"
    ofile="${OUTDIR}/${PREFIX}.d$(printf "%02d" $i)_of_$(printf "%02d" $NJOBS).slurm"
    make_one_diam "$i" "$mind" "$maxd" "$ofile"
    outfiles+=( "$ofile" )
    if [[ -n "$COUNT_FILE" ]]; then
      nbin=$(count_in_range "$mind" "$maxd" "$COUNT_FILE")
      printf "  [%2d/%2d] %-40s  --mindiam=%.5f  --maxdiam=%.5f     [%d objects]\n" \
        "$((i+1))" "$NJOBS" "$(basename "$ofile")" "$mind" "$maxd" "$nbin"
    else
      printf "  [%2d/%2d] %-40s  --mindiam=%.5f  --maxdiam=%.5f\n" \
        "$((i+1))" "$NJOBS" "$(basename "$ofile")" "$mind" "$maxd"
    fi
  done

elif [[ "$SPACING" == "linear" ]]; then
  step=$(awk -v a="$MIN" -v b="$MAX" -v n="$NJOBS" 'BEGIN{ printf "%.17g", (b-a)/n }')
  printf "Generating %d jobs (linear) over [%.5f, %.5f] step %.5f\n" \
         "$NJOBS" "$(fmt5 "$MIN")" "$(fmt5 "$MAX")" "$(fmt5 "$step")"

  for (( i=0; i<NJOBS; i++ )); do
    mind=$(awk -v a="$MIN" -v s="$step" -v i="$i" 'BEGIN{ printf "%.17g", a + i*s }')
    if (( i == NJOBS-1 )); then
      maxd="$MAX"
    else
      maxd=$(awk -v a="$MIN" -v s="$step" -v i="$i" 'BEGIN{ printf "%.17g", a + (i+1)*s }')
    fi
    ofile="${OUTDIR}/${PREFIX}.d$(printf "%02d" $i)_of_$(printf "%02d" $NJOBS).slurm"
    make_one_diam "$i" "$mind" "$maxd" "$ofile"
    outfiles+=( "$ofile" )
    if [[ -n "$COUNT_FILE" ]]; then
      nbin=$(count_in_range "$mind" "$maxd" "$COUNT_FILE")
      printf "  [%2d/%2d] %-40s  --mindiam=%s  --maxdiam=%s     [%d objects]\n" \
             "$((i+1))" "$NJOBS" "$(basename "$ofile")" "$(fmt5 "$mind")" "$(fmt5 "$maxd")" "$nbin"
    else
      printf "  [%2d/%2d] %-40s  --mindiam=%s  --maxdiam=%s\n" \
             "$((i+1))" "$NJOBS" "$(basename "$ofile")" "$(fmt5 "$mind")" "$(fmt5 "$maxd")"
    fi
  done

else
  ln10=$(awk 'BEGIN{ print log(10) }')
  log10() { awk -v x="$1" -v l10="$ln10" 'BEGIN{ printf "%.17g", log(x)/l10 }'; }
  pow10() { awk -v y="$1" -v l10="$ln10" 'BEGIN{ printf "%.17g", exp(y*l10) }'; }

  lo=$(log10 "$MIN"); hi=$(log10 "$MAX")
  dlog=$(awk -v a="$lo" -v b="$hi" -v n="$NJOBS" 'BEGIN{ printf "%.17g", (b-a)/n }')
  echo "Generating $NJOBS jobs (log10) over [$(fmt5 "$MIN"), $(fmt5 "$MAX")]"

  for (( i=0; i<NJOBS; i++ )); do
    l1=$(awk -v a="$lo" -v s="$dlog" -v i="$i" 'BEGIN{ printf "%.17g", a + i*s }')
    if (( i == NJOBS-1 )); then
      l2="$hi"
    else
      l2=$(awk -v a="$lo" -v s="$dlog" -v i="$i" 'BEGIN{ printf "%.17g", a + (i+1)*s }')
    fi
    mind=$(pow10 "$l1")
    maxd=$(pow10 "$l2")

    ofile="${OUTDIR}/${PREFIX}.d$(printf "%02d" $i)_of_$(printf "%02d" $NJOBS).slurm"
    make_one_diam "$i" "$mind" "$maxd" "$ofile"
    outfiles+=( "$ofile" )
    if [[ -n "$COUNT_FILE" ]]; then
      nbin=$(count_in_range "$mind" "$maxd" "$COUNT_FILE")
      printf "  [%2d/%2d] %-40s  --mindiam=%s  --maxdiam=%s     [%d objects]\n" \
             "$((i+1))" "$NJOBS" "$(basename "$ofile")" "$(fmt5 "$mind")" "$(fmt5 "$maxd")" "$nbin"
    else
      printf "  [%2d/%2d] %-40s  --mindiam=%s  --maxdiam=%s\n" \
             "$((i+1))" "$NJOBS" "$(basename "$ofile")" "$(fmt5 "$mind")" "$(fmt5 "$maxd")"
    fi
  done
fi

# ------------------ Optional submission ------------------
if (( SUBMIT )); then
  echo "Submitting ${#outfiles[@]} jobs with sbatch..."
  for (( i=${#outfiles[@]}-1; i>=0; i-- )); do
    sbatch "${outfiles[i]}"
  done
fi
