#!/usr/bin/env bash
set -euo pipefail

# Build the official GaussianImage source and its pinned gsplat submodule in an
# isolated, reproducible environment. GaussianImage itself does not publish an
# environment file; the selected Torch/CUDA pair is one of the combinations in
# the pinned gsplat commit's wheel-build matrix.
ROOT="$(realpath -m "${1:-${PWD}/results/native_envs/gaussianimage_official}")"
REPO_DIR="${ROOT}/repo"
ENV_PREFIX="${ROOT}/env"
PROVENANCE_DIR="${ROOT}/provenance"
WHEEL_DIR="${ROOT}/wheels"

GAUSSIANIMAGE_URL="https://github.com/Xinjie-Q/GaussianImage.git"
GAUSSIANIMAGE_COMMIT="d53393bee7c9fbb24e3510614e3ff2c85b8fbbc1"
GSPLAT_COMMIT="bcca3ecae966a052e3bf8dd1ff9910cf7b8f851d"
CUDA_LABEL="nvidia/label/cuda-11.8.0"
CUDA_VERSION="11.8"
TORCH_VERSION="2.0.0"
TORCHVISION_VERSION="0.15.1"
SYSTEM_LIBSTDCXX="/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

if [[ ! -e "${SYSTEM_LIBSTDCXX}" ]]; then
  echo "System libstdc++ preload not found: ${SYSTEM_LIBSTDCXX}" >&2
  exit 2
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to provision the isolated native environment" >&2
  exit 2
fi
if [[ ! -x /usr/bin/gcc-11 || ! -x /usr/bin/g++-11 ]]; then
  echo "CUDA 11.8 requires the installed GCC/G++ 11 host compiler" >&2
  exit 2
fi

mkdir -p "${ROOT}" "${PROVENANCE_DIR}" "${WHEEL_DIR}"

CLONED_REPO=0
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --no-checkout --filter=blob:none "${GAUSSIANIMAGE_URL}" "${REPO_DIR}"
  CLONED_REPO=1
fi
if [[ "${CLONED_REPO}" -eq 0 && -n "$(git -C "${REPO_DIR}" status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing to alter a tracked-dirty isolated checkout: ${REPO_DIR}" >&2
  exit 2
fi
git -C "${REPO_DIR}" fetch origin "${GAUSSIANIMAGE_COMMIT}"
git -C "${REPO_DIR}" checkout --detach "${GAUSSIANIMAGE_COMMIT}"
git -C "${REPO_DIR}" submodule update --init --recursive

ACTUAL_REPO_COMMIT="$(git -C "${REPO_DIR}" rev-parse HEAD)"
ACTUAL_GSPLAT_COMMIT="$(git -C "${REPO_DIR}/gsplat" rev-parse HEAD)"
if [[ "${ACTUAL_REPO_COMMIT}" != "${GAUSSIANIMAGE_COMMIT}" ]]; then
  echo "GaussianImage commit mismatch: ${ACTUAL_REPO_COMMIT}" >&2
  exit 2
fi
if [[ "${ACTUAL_GSPLAT_COMMIT}" != "${GSPLAT_COMMIT}" ]]; then
  echo "gsplat commit mismatch: ${ACTUAL_GSPLAT_COMMIT}" >&2
  exit 2
fi

if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  conda create --yes --prefix "${ENV_PREFIX}" python=3.10 pip=24.2 setuptools=69.5.1 wheel=0.43.0
fi
conda install --yes --prefix "${ENV_PREFIX}" --override-channels \
  --channel "${CUDA_LABEL}" --channel defaults \
  "cuda-nvcc=${CUDA_VERSION}" "cuda-cudart-dev=${CUDA_VERSION}"

PYTHONNOUSERSITE=1 "${ENV_PREFIX}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu118 \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# GaussianImage's requirements file is unpinned. Resolve it at the source
# commit's 2024-08-26 cutoff; in particular, current vector-quantize-pytorch
# requires a newer Torch and would silently replace the gsplat-supported stack.
# NumPy remains at the latest 1.x release because Torch 2.0 predates NumPy 2's
# ABI. The complete transitive resolution is captured below with pip freeze.
PYTHONNOUSERSITE=1 "${ENV_PREFIX}/bin/python" -m pip install \
  "constriction==0.3.5" \
  "numpy==1.26.4" \
  "pandas==2.2.2" \
  "pillow==10.4.0" \
  "pytorch-msssim==1.0.0" \
  "PyYAML==6.0.2" \
  "tqdm==4.66.5" \
  "vector-quantize-pytorch==1.16.1"

# Build from an archive of the exact submodule commit so setup.py cannot leave
# generated files in the provenance checkout. Keep the wheel as a hashed build
# artifact and then install it without allowing dependency substitution.
BUILD_ROOT="$(mktemp -d "${ROOT}/gsplat-build.XXXXXX")"
cleanup() {
  rm -rf "${BUILD_ROOT}"
}
trap cleanup EXIT
git -C "${REPO_DIR}/gsplat" archive HEAD | tar -x -C "${BUILD_ROOT}"

# Match the upstream .[dev] installation instructions while keeping extension
# compilation in the explicit wheel-build step below.
BUILD_NO_CUDA=1 PYTHONNOUSERSITE=1 "${ENV_PREFIX}/bin/python" -m pip install \
  --no-build-isolation "${BUILD_ROOT}[dev]"
rm -f "${WHEEL_DIR}"/gsplat-*.whl
CC=/usr/bin/gcc-11 \
CXX=/usr/bin/g++-11 \
CUDA_HOME="${ENV_PREFIX}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
PYTHONNOUSERSITE=1 \
"${ENV_PREFIX}/bin/python" -m pip wheel \
  --no-build-isolation --no-deps --wheel-dir "${WHEEL_DIR}" "${BUILD_ROOT}"
PYTHONNOUSERSITE=1 "${ENV_PREFIX}/bin/python" -m pip install \
  --force-reinstall --no-deps "${WHEEL_DIR}"/gsplat-*.whl

export GAUSSIANIMAGE_NATIVE_ROOT="${ROOT}"
export GAUSSIANIMAGE_NATIVE_REPO="${REPO_DIR}"
export GAUSSIANIMAGE_NATIVE_PRELOAD="${SYSTEM_LIBSTDCXX}"
LD_PRELOAD="${SYSTEM_LIBSTDCXX}" \
PYTHONNOUSERSITE=1 \
PYTHONDONTWRITEBYTECODE=1 \
"${ENV_PREFIX}/bin/python" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

root = Path(os.environ["GAUSSIANIMAGE_NATIVE_ROOT"])
repo = Path(os.environ["GAUSSIANIMAGE_NATIVE_REPO"])
preload = Path(os.environ["GAUSSIANIMAGE_NATIVE_PRELOAD"])
provenance = root / "provenance"

import torch
import torchvision
import gsplat
import gsplat.csrc as csrc

sys.path.insert(0, str(repo))
from gaussianimage_cholesky import GaussianImage_Cholesky

device = torch.device("cuda:0")
torch.manual_seed(0)
model = GaussianImage_Cholesky(
    loss_type="L2",
    num_points=4,
    H=16,
    W=16,
    BLOCK_W=16,
    BLOCK_H=16,
    device=device,
    quantize=False,
    opt_type="adam",
    lr=1e-3,
).to(device)
render = model()["render"]
render.sum().backward()
torch.cuda.synchronize()

csrc_path = Path(csrc.__file__).resolve()
wheel_paths = sorted((root / "wheels").glob("gsplat-*.whl"))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

report = {
    "gaussianimage_commit": subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip(),
    "gsplat_commit": subprocess.check_output(
        ["git", "-C", str(repo / "gsplat"), "rev-parse", "HEAD"], text=True
    ).strip(),
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "torchvision": torchvision.__version__,
    "gsplat": importlib.metadata.version("gsplat"),
    "gsplat_python": str(Path(gsplat.__file__).resolve()),
    "gsplat_csrc": str(csrc_path),
    "gsplat_csrc_sha256": sha256(csrc_path),
    "wheel_sha256": {path.name: sha256(path) for path in wheel_paths},
    "system_libstdcxx": str(preload.resolve()),
    "system_libstdcxx_sha256": sha256(preload.resolve()),
    "loaded_libstdcxx": sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines()
            if "libstdc++.so" in line
        }
    ),
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0),
    "cuda_capability": list(torch.cuda.get_device_capability(0)),
    "smoke_render_shape": list(render.shape),
    "smoke_render_finite": bool(torch.isfinite(render).all().item()),
    "smoke_backward": all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    ),
}
(provenance / "verify.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

git -C "${REPO_DIR}" rev-parse HEAD > "${PROVENANCE_DIR}/gaussianimage-commit.txt"
git -C "${REPO_DIR}" submodule status --recursive > "${PROVENANCE_DIR}/submodules.txt"
git -C "${REPO_DIR}" status --porcelain=v1 > "${PROVENANCE_DIR}/repo-status.txt"
git -C "${REPO_DIR}" ls-files -s > "${PROVENANCE_DIR}/gaussianimage-index.txt"
git -C "${REPO_DIR}/gsplat" ls-files -s > "${PROVENANCE_DIR}/gsplat-index.txt"
sha256sum "${REPO_DIR}/requirements.txt" "${REPO_DIR}/.gitmodules" \
  > "${PROVENANCE_DIR}/source-file-sha256.txt"
sha256sum "${WHEEL_DIR}"/gsplat-*.whl > "${PROVENANCE_DIR}/wheel-sha256.txt"
sha256sum "${SYSTEM_LIBSTDCXX}" > "${PROVENANCE_DIR}/system-libstdcxx-sha256.txt"
PYTHONNOUSERSITE=1 "${ENV_PREFIX}/bin/python" -m pip freeze --all \
  > "${PROVENANCE_DIR}/pip-freeze.txt"
PYTHONNOUSERSITE=1 "${ENV_PREFIX}/bin/python" -m pip check \
  > "${PROVENANCE_DIR}/pip-check.txt"
conda list --prefix "${ENV_PREFIX}" --explicit > "${PROVENANCE_DIR}/conda-explicit.txt"
conda list --prefix "${ENV_PREFIX}" --json > "${PROVENANCE_DIR}/conda-list.json"
{
  uname -a
  /usr/bin/gcc-11 --version
  /usr/bin/g++-11 --version
  "${ENV_PREFIX}/bin/nvcc" --version
  nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
  LD_PRELOAD="${SYSTEM_LIBSTDCXX}" \
    LD_LIBRARY_PATH="${ENV_PREFIX}/lib/python3.10/site-packages/torch/lib:${ENV_PREFIX}/lib" \
    ldd "$(find "${ENV_PREFIX}" -path '*/site-packages/gsplat/csrc.so' -print -quit)"
} > "${PROVENANCE_DIR}/system-and-linkage.txt"

echo "GaussianImage native environment ready: ${ENV_PREFIX}"
echo "Pinned source checkout: ${REPO_DIR}"
echo "Verification report: ${PROVENANCE_DIR}/verify.json"
echo "Run with LD_PRELOAD=${SYSTEM_LIBSTDCXX} and PYTHONNOUSERSITE=1"
