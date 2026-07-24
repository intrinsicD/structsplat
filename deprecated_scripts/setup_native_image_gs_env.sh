#!/usr/bin/env bash
set -euo pipefail

# Reproduce the official Image-GS dependency versions while pinning the formerly floating
# fused-ssim dependency to the last commit preceding the Image-GS release.
IMAGE_GS_REPO="${1:-/home/alex/Documents/image-gs}"
ENV_PREFIX="${2:-${PWD}/results/native_envs/image_gs_official}"
FUSED_SSIM_COMMIT="b4fd8324e81c48c9b2b9f62e1b9c6431fece6ab3"
# PyTorch 2.4.1's conda build requires the Intel JIT notification symbols that
# disappeared from newer MKL packages.  The upstream environment did not pin
# this transitive dependency, so a current solve selects MKL 2025 and torch
# fails to import with an undefined iJIT_NotifyEvent symbol.
MKL_VERSION="2023.1.0"
# Current NVIDIA repodata otherwise lets pytorch-cuda=12.4 resolve against a
# cuda-version=13.3 marker and 13.3 ancillary packages.
CUDA_VERSION="12.4"

if [[ ! -f "${IMAGE_GS_REPO}/environment.yml" ]]; then
  echo "Image-GS environment file not found: ${IMAGE_GS_REPO}/environment.yml" >&2
  exit 2
fi

RESOLVED_ENV_FILE="$(mktemp --suffix=.yml)"
trap 'rm -f "${RESOLVED_ENV_FILE}"' EXIT
awk -v mkl_version="${MKL_VERSION}" -v cuda_version="${CUDA_VERSION}" '
  /^  - pip:/ && !injected {
    print "  - mkl=" mkl_version
    print "  - cuda-version=" cuda_version
    injected = 1
  }
  { print }
  END { if (!injected) exit 2 }
' "${IMAGE_GS_REPO}/environment.yml" > "${RESOLVED_ENV_FILE}"

conda env create --prefix "${ENV_PREFIX}" --file "${RESOLVED_ENV_FILE}"
PYTHONNOUSERSITE=1 "${ENV_PREFIX}/bin/python" -m pip install \
  --no-build-isolation --no-deps \
  "git+https://github.com/rahul-goel/fused-ssim/@${FUSED_SSIM_COMMIT}"
PYTHONNOUSERSITE=1 "${ENV_PREFIX}/bin/python" -m pip install \
  --no-build-isolation "${IMAGE_GS_REPO}/gsplat[dev]"

echo "Image-GS native environment ready: ${ENV_PREFIX}"
echo "Run with LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 if csrc import needs it."
