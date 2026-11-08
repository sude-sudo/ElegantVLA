temporal_skip_resolve_repo_root() {
    candidate=$(CDPATH= cd -- "$1" && pwd)
    while [ ! -d "$candidate/elegantvla/gr00t" ] || [ ! -d "$candidate/third_party/Isaac-GR00T" ]; do
        if [ "$candidate" = "/" ]; then
            echo "Could not locate the ElegantVLA source tree." >&2
            return 1
        fi
        candidate=$(dirname -- "$candidate")
    done
    printf '%s\n' "$candidate"
}


temporal_skip_setup_python_env() {
    repo_root=$1
    default_mplconfigdir=$2
    gr00t_root="$repo_root/third_party/Isaac-GR00T"
    pythonpath_prefix="$repo_root:$gr00t_root"
    simplerenv_source_root=${SIMPLERENV_ROOT:-$gr00t_root/external_dependencies/SimplerEnv}

    if [ -n "${SIMPLERENV_ROOT:-}" ] && [ ! -d "$SIMPLERENV_ROOT/simpler_env" ]; then
        echo "SIMPLERENV_ROOT must contain the simpler_env package." >&2
        return 1
    fi
    if [ -d "$simplerenv_source_root/simpler_env" ]; then
        pythonpath_prefix="$pythonpath_prefix:$simplerenv_source_root:$simplerenv_source_root/ManiSkill2_real2sim"
    fi
    if [ -n "${SIMPLERENV_SITE_PACKAGES:-}" ]; then
        if [ ! -d "$SIMPLERENV_SITE_PACKAGES" ]; then
            echo "SIMPLERENV_SITE_PACKAGES does not exist." >&2
            return 1
        fi
        pythonpath_prefix="$pythonpath_prefix:$SIMPLERENV_SITE_PACKAGES"
    fi

    export PYTHONPATH="$pythonpath_prefix${PYTHONPATH:+:$PYTHONPATH}"
    export ELEGANTVLA_ROOT="$repo_root"
    export TEMPORAL_SKIP_REPO_ROOT="$gr00t_root"
    export MPLCONFIGDIR=${MPLCONFIGDIR:-$default_mplconfigdir}
    cpu_threads=${TEMPORAL_SKIP_CPU_THREADS:-1}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$cpu_threads}
    export MKL_NUM_THREADS=${MKL_NUM_THREADS:-$cpu_threads}
    export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-$cpu_threads}
    export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-$cpu_threads}
    export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-4}
    mkdir -p "$MPLCONFIGDIR"
    cd "$repo_root"
}


temporal_skip_exec_module() {
    python_bin=$1
    module_name=$2
    shift 2
    "$python_bin" -m "$module_name" "$@"
}


temporal_skip_exec_file() {
    python_bin=$1
    script_path=$2
    shift 2
    "$python_bin" "$script_path" "$@"
}
