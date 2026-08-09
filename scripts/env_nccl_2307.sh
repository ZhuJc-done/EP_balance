#!/usr/bin/env bash
# Force every child Python process to load the validated NCCL 2.30.7 wheel.

_eplb_nccl_version="$(python -c 'import importlib.metadata as m; print(m.version("nvidia-nccl-cu13"))' 2>/dev/null || true)"
if [[ "${_eplb_nccl_version}" != "2.30.7" ]]; then
  echo "expected nvidia-nccl-cu13==2.30.7, found ${_eplb_nccl_version:-<missing>}" >&2
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 1; else exit 1; fi
fi

_eplb_nccl_root="$(python -c 'import nvidia.nccl as n; print(n.__path__[0])')"
_eplb_nccl_so="${_eplb_nccl_root}/lib/libnccl.so.2"
if [[ ! -f "${_eplb_nccl_so}" ]]; then
  echo "NCCL 2.30.7 runtime is missing: ${_eplb_nccl_so}" >&2
  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 1; else exit 1; fi
fi

_eplb_preload=""
_eplb_existing_preload="${LD_PRELOAD:-}"
for _eplb_lib in ${_eplb_existing_preload//:/ }; do
  [[ "${_eplb_lib##*/}" == libnccl.so* ]] && continue
  _eplb_preload="${_eplb_preload:+${_eplb_preload}:}${_eplb_lib}"
done

export NCCL_HOME="${_eplb_nccl_root}"
export EP_NCCL_ROOT_DIR="${_eplb_nccl_root}"
export LD_PRELOAD="${_eplb_nccl_so}${_eplb_preload:+:${_eplb_preload}}"
export LD_LIBRARY_PATH="${_eplb_nccl_root}/lib:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${_eplb_nccl_root}/lib:${LIBRARY_PATH:-}"
export EP_REUSE_NCCL_COMM=0

unset _eplb_nccl_version _eplb_nccl_root _eplb_nccl_so _eplb_preload
unset _eplb_existing_preload _eplb_lib
