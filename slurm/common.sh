# Shared by every job and by the submit / status scripts. Not run directly.
# Jobs get RUN_ENV (the frozen settings written by submit_all.sh) through sbatch --export.

# ---------------------------------------------------------------- settings
# REPO and the settings: from the frozen run.env inside a job, from settings.sh otherwise.
load_settings() {
    if [[ -n "${RUN_ENV:-}" ]]; then
        # shellcheck disable=SC1090
        source "$RUN_ENV"
    else
        REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
        # shellcheck source=settings.sh
        source "$REPO/slurm/settings.sh"
    fi
    [[ -n "${ROOT:-}" && "$ROOT" != "/superres" ]] || die "ROOT is empty: is \$SCRATCH set? (see settings.sh)"
    read -r -a LEVEL_LIST <<< "$LEVELS"
    read -r -a SEED_LIST <<< "$SEEDS"
    ((${#LEVEL_LIST[@]})) || die "LEVELS is empty"
    ((${#SEED_LIST[@]})) || die "SEEDS is empty"
}

die() { echo "ERROR: $*" >&2; exit 1; }

# For scripts run by hand after submission: load the frozen settings of one experiment.
#   load_experiment ""         the most recently submitted one
#   load_experiment <name>     that one, e.g. 2026-10-06_v2_multiseed
#   load_experiment --smoke    the most recent smoke test
load_experiment() {
    local logs="$ROOT/logs" id="${1:-}" env_file=""
    if [[ "$id" == --smoke ]]; then
        logs="$ROOT/smoke/logs"
        id=""
    fi
    env_file="$(newest_env_file "$logs" "${id:-*}")"
    [[ -f "$env_file" ]] || die "no submitted $DATASET experiment found under $logs${id:+ for $id}"
    RUN_ENV="$env_file"
    load_settings
}

# Newest run.env of a $DATASET experiment whose name matches a pattern (empty if none). Looks in
# logs/<dataset>/<experiment>/ and, for experiments submitted before logs were split by dataset,
# in logs/<experiment>/ (only if that run.env is of this dataset).
newest_env_file() {
    local logs="$1" pattern="$2" f
    local -a files=()
    # shellcheck disable=SC2086
    for f in "$logs/$DATASET/"$pattern/run.env "$logs/"$pattern/run.env; do
        [[ -f "$f" ]] && grep -q "^export DATASET=$DATASET\$" "$f" && files+=("$f")
    done
    ((${#files[@]})) || return 0
    # shellcheck disable=SC2012
    ls -t "${files[@]}" | head -1 || true
}

# Most recent experiment with this dataset, version and tag -> its date (empty if none).
latest_experiment_date() {
    local f
    f="$(newest_env_file "$ROOT/logs" "*_${VERSION}_${RUN_TAG}")"
    [[ -n "$f" ]] && basename "$(dirname "$f")" | cut -c1-10
    return 0
}

# Logs and frozen settings of the current experiment. Split by dataset, so a qg and a calo experiment
# with the same date and tag never share run.env. Experiments submitted before the split keep their folder.
log_dir() {
    local old="$ROOT/logs/$EXPERIMENT"
    if [[ ! -d "$ROOT/logs/$DATASET/$EXPERIMENT" && -f "$old/run.env" ]] \
        && grep -q "^export DATASET=$DATASET\$" "$old/run.env"; then
        echo "$old"
    else
        echo "$ROOT/logs/$DATASET/$EXPERIMENT"
    fi
}

# Fix the experiment name once (submit_all.sh): RUN_DATE (today if empty), RUN_TAG (+ suffix) and
# EXPERIMENT = <date>_<version>_<tag>, the same rule as superres.io_utils.experiment_name.
resolve_experiment() {
    RUN_DATE="${RUN_DATE:-$(TZ=Asia/Kolkata date +%Y-%m-%d)}"
    RUN_TAG="${RUN_TAG}${1:-}"
    [[ "$RUN_DATE" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || die "RUN_DATE must look like 2026-10-06, not '$RUN_DATE'"
    [[ "$RUN_TAG" =~ ^[A-Za-z0-9.-]+$ ]] || die "RUN_TAG may only use letters, digits, '-' and '.': '$RUN_TAG'"
    EXPERIMENT="${RUN_DATE}_${VERSION}_${RUN_TAG}"
}

experiment_dir() { echo "$ROOT/experiments/$DATASET/$EXPERIMENT"; }
prefix() { echo "${DATASET}_${VERSION}_${RUN_DATE}_${RUN_TAG}"; }    # start of every file name

# ---------------------------------------------------------------- work lists
# Tagger inputs, split so that the ones not needing SR can start right after prepare.
#   early: hr and lr:<level>      late: sr:<level>
tagger_kinds() {
    local group="$1" l
    if [[ "$group" == early ]]; then
        echo hr
        for l in "${LEVEL_LIST[@]}"; do echo "lr:$l"; done
    else
        for l in "${LEVEL_LIST[@]}"; do echo "sr:$l"; done
        # calogan: the k-NN template baseline is written by the generate job
        if [[ "$DATASET" == calogan ]]; then
            for l in "${LEVEL_LIST[@]}"; do echo "template:$l"; done
        fi
    fi
}

# Line i (0-based) = "<kind> <seed>" for array task i.
tagger_tasks() {
    local k s
    while read -r k; do
        for s in "${SEED_LIST[@]}"; do echo "$k $s"; done
    done < <(tagger_kinds "$1")
}

tagger_task() { tagger_tasks "$1" | sed -n "$(($2 + 1))p"; }
n_tagger_tasks() { tagger_tasks "$1" | wc -l | tr -d ' '; }

# ---------------------------------------------------------------- environment
setup_python() {
    if ! type module > /dev/null 2>&1 && [[ -f /usr/share/lmod/lmod/init/bash ]]; then
        # shellcheck disable=SC1091
        source /usr/share/lmod/lmod/init/bash
    fi
    if type module > /dev/null 2>&1; then
        set +u    # Lmod's shell code reads unset variables
        module load pytorch
        set -u
    fi
    export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    export MPLBACKEND=Agg
    local cpus="${SLURM_CPUS_PER_TASK:-4}"
    PHYS_CORES=$((cpus > 1 ? cpus / 2 : 1))   # Perlmutter: 2 hardware threads per core
    export OMP_NUM_THREADS="$PHYS_CORES"
}

job_header() {
    echo "==================================================================="
    echo "stage      : $1"
    echo "experiment : $(experiment_dir)"
    echo "job        : ${SLURM_JOB_ID:-local} task ${SLURM_ARRAY_TASK_ID:--} on $(hostname)"
    echo "started    : $(date '+%Y-%m-%d %H:%M:%S %Z')"
    echo "code       : $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2> /dev/null) $(git -C "$REPO" rev-parse --short HEAD 2> /dev/null)"
    echo "settings   : dataset=$DATASET version=$VERSION levels='$LEVELS' seeds='$SEEDS' qos=$QOS extra='$EXTRA_SET'"
    python -c 'import sys, torch; print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda", torch.cuda.is_available())'
    # CPU nodes (prepare) may have nvidia-smi on PATH but no driver: it then fails, and under
    # set -e that failure would end the job here, before python starts
    if command -v nvidia-smi > /dev/null; then
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2> /dev/null || echo "(no GPU on this node)"
    fi
    echo "==================================================================="
}

# run_pipeline <stages> [--levels L] [--kinds K] [--seeds S] [--smoke]
# Runs python -m superres.pipeline with this experiment's paths; records the session in run_info.json.
# One process per job, started directly in the batch step so it gets all the job's CPUs and its GPU
# (an srun step would not inherit -c from sbatch on current Slurm versions).
run_pipeline() {
    local stages="$1"
    shift
    local workers=$((PHYS_CORES > 2 ? PHYS_CORES / 2 : 1))
    local -a sets=("paths.drive_root=$ROOT/experiments" "paths.raw_root=$ROOT/raw" "paths.cache_root=$ROOT/cache"
                   "version=$VERSION" "run_date=$RUN_DATE" "run_tag=$RUN_TAG" "num_workers=$workers")
    local -a extra=()
    [[ -n "$EXTRA_SET" ]] && read -r -a extra <<< "$EXTRA_SET"
    local -a cmd=(python -m superres.pipeline --dataset "$DATASET" --stages "$stages" "$@" --run-info
                  --set "${sets[@]}" ${extra[@]+"${extra[@]}"})
    echo "+ ${cmd[*]}"
    local t0=$SECONDS
    "${cmd[@]}"
    echo "-- $stages finished in $(((SECONDS - t0) / 60)) min"
}
