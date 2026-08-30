#!/usr/bin/env bash
# 安装本项目 multitask 体系所需的 .venv 环境与依赖。
#
# 功能：
#   1. 用 uv 在仓库根目录创建 .venv（Python 3.12，可用 PYTHON_VERSION 覆盖）；
#      已存在的 .venv 默认复用，仅当 Python 版本不符时才清空重建；
#   2. 自动检测 NVIDIA 驱动版本，选择匹配的 PyTorch CUDA wheel 索引
#      （无 GPU、驱动过旧或检测失败时退回 CPU wheel）；
#   3. 其余依赖以 pyproject.toml 为唯一来源安装（含 timm git pin，
#      tyro / wandb / lerobot / torchcodec / av / pyarrow / h5py 等），
#      不再在脚本内维护第二份依赖清单；
#   4. 默认从官方 PyPI / PyTorch 索引安装；--use-mirror 时改用阿里云 PyPI 镜像
#      （阿里云为全量同步镜像，含 cyclonedds-nightly 等 nightly 包；
#       清华/中科大/腾讯/网易镜像未同步该类包）
#      与上海交大 PyTorch wheel 镜像（可用 PYPI_INDEX / PYTORCH_INDEX_BASE 覆盖）；
#   5. xensesdk 为私有可选包：先尝试 uv pip install，失败则退回 pip 安装，
#      再失败仅告警不中断（仅 scripts/build_labeled_dataset.py annotate 阶段需要）。
#
# 用法：
#   bash install.sh                       # 普通安装（官方 PyPI / PyTorch 索引）
#   bash install.sh --use-mirror          # 走镜像安装（阿里云 PyPI + 上海交大 PyTorch wheel）
#   TORCH_CUDA=cu128 bash install.sh      # 手动指定 CUDA tag（cu130/cu128/cpu）
#   SKIP_XENSESDK=1 bash install.sh       # 跳过 xensesdk
set -euo pipefail

log() { printf '[install] %s\n' "$*"; }

usage() {
    cat <<'EOF'
用法：bash install.sh [选项]

选项：
  --use-mirror   使用国内镜像安装：PyPI 走阿里云 mirrors.aliyun.com（全量同步，
                 含 nightly 包；清华/中科大/腾讯/网易缺 cyclonedds-nightly），
                 PyTorch wheel 索引走上海交大 mirror.sjtu.edu.cn（官方
                 download.pytorch.org 在国内网络常不可达）；
                 分别可用 PYPI_INDEX / PYTORCH_INDEX_BASE 覆盖
  -h, --help     显示本帮助

环境变量：
  TORCH_CUDA     CUDA wheel tag（cu130 / cu128 / cpu），默认自动检测
  PYTHON_VERSION Python 版本，默认 3.12
  VENV_DIR       虚拟环境路径，默认 <仓库根>/.venv
  PYPI_INDEX     PyPI 索引地址（覆盖镜像默认值；普通安装时也生效）
  PYTORCH_INDEX_BASE  PyTorch wheel 索引基地址（覆盖镜像默认值；普通安装时也生效）
  SKIP_XENSESDK  设为 1 跳过 xensesdk 安装
EOF
}

# --- 参数解析 ---------------------------------------------------------------
USE_MIRROR=0
for arg in "$@"; do
    case "${arg}" in
        --use-mirror) USE_MIRROR=1 ;;
        -h|--help) usage; exit 0 ;;
        *) log "错误：未知参数：${arg}"; usage; exit 1 ;;
    esac
done

# --- 定位仓库根（脚本可位于仓库根或 scripts/ 子目录） ------------------------
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${REPO_ROOT}" || ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    # 非 git 场景兜底：从脚本所在目录向上找含 pyproject.toml 的目录
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    while [[ "${REPO_ROOT}" != "/" && ! -f "${REPO_ROOT}/pyproject.toml" ]]; do
        REPO_ROOT="$(dirname "${REPO_ROOT}")"
    done
fi
if [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
    log "错误：找不到仓库根（未找到 pyproject.toml），请确认 install.sh 位于仓库内"
    exit 1
fi
log "仓库根：${REPO_ROOT}"

VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.25.0}"
PYPI_INDEX="${PYPI_INDEX:-}"   # 空 = 官方 PyPI（uv 默认）
PYPI_INDEX_MIRROR="https://mirrors.aliyun.com/pypi/simple/"
# 国内 PyTorch wheel 镜像：仅此一家提供 PEP 503 索引（清华/阿里/华为/中科大均无）。
# 注意：索引经 SJTU，但 wheel 文件本身仍从官方 CDN download-r2.pytorch.org 下载。
PYTORCH_INDEX_BASE_MIRROR="https://mirror.sjtu.edu.cn/pytorch-wheels"
PYTORCH_INDEX_BASE="${PYTORCH_INDEX_BASE:-}"   # 空 = 官方 https://download.pytorch.org/whl

if [[ "${USE_MIRROR}" == "1" ]]; then
    PYPI_INDEX="${PYPI_INDEX:-${PYPI_INDEX_MIRROR}}"
    PYTORCH_INDEX_BASE="${PYTORCH_INDEX_BASE:-${PYTORCH_INDEX_BASE_MIRROR}}"
    log "使用镜像安装：PyPI=${PYPI_INDEX}"
else
    log "使用官方源安装${PYPI_INDEX:+（PYPI_INDEX=${PYPI_INDEX}）}"
fi
# 最终默认：未显式指定时，普通安装用官方索引
PYTORCH_INDEX_BASE="${PYTORCH_INDEX_BASE:-https://download.pytorch.org/whl}"

if ! command -v uv >/dev/null 2>&1; then
    log "错误：未找到 uv，请先安装（curl -LsSf https://astral.sh/uv/install.sh | sh）"
    exit 1
fi

# --- 1. 创建/复用 .venv（仅 Python 版本不符时才重建） ------------------------
ensure_venv() {
    if [[ -x "${VENV_DIR}/bin/python" ]]; then
        local existing want
        existing="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
        want="${PYTHON_VERSION}"
        [[ "${want}" == *.*.* ]] && want="${want%.*}"
        if [[ -n "${existing}" && "${existing}" == "${want}" ]]; then
            log "复用已有虚拟环境 ${VENV_DIR}（Python ${existing}），不重建"
            return
        fi
        log "警告：${VENV_DIR} 的 Python（${existing:-未知}）与要求 ${PYTHON_VERSION} 不符，将清空重建"
        uv venv --clear --python "${PYTHON_VERSION}" "${VENV_DIR}"
        return
    fi
    log "创建虚拟环境 ${VENV_DIR} (Python ${PYTHON_VERSION})"
    uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
}
ensure_venv

# --- 2. 检测驱动版本，选择 torch wheel ---------------------------------------
# cu130 wheel 要求驱动 >= 580（对应 CUDA 13.0）；cu128 要求驱动 >= 570（CUDA 12.8）。
detect_cuda_tag() {
    # 输出 cu130 / cu128 / cpu；可被 TORCH_CUDA 环境变量覆盖。
    if [[ -n "${TORCH_CUDA:-}" ]]; then
        echo "${TORCH_CUDA}"
        return
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "cpu"
        return
    fi
    local driver_version major
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
    driver_version="${driver_version%% *}"
    if [[ ! "${driver_version}" =~ ^[0-9]+(\.[0-9]+)*$ ]]; then
        echo "cpu"
        return
    fi
    major="${driver_version%%.*}"
    if (( major >= 580 )); then
        echo "cu130"
    elif (( major >= 570 )); then
        echo "cu128"
    else
        log "警告：检测到 NVIDIA 驱动 ${driver_version}，低于 cu128 wheel 所需的最低版本 570（CUDA 12.8）"
        log "       将退回 CPU 构建；如需 GPU 请升级驱动，或手动指定 TORCH_CUDA=cu128 等"
        echo "cpu"
    fi
}

CUDA_TAG="$(detect_cuda_tag)"
DRIVER_VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
log "检测结果：CUDA tag = ${CUDA_TAG}（驱动版本: ${DRIVER_VER:-无}）"
log "PyTorch wheel 索引：${PYTORCH_INDEX_BASE}/${CUDA_TAG}"

# --- 3. 安装 torch / torchvision（已装目标版本则跳过） ---------------------------
DEP_ARGS=()
[[ -n "${PYPI_INDEX}" ]] && DEP_ARGS+=(--index-url "${PYPI_INDEX}")

installed_pkg_version() {
    # 查询 .venv 内已装包的版本（含 +cuXXX 本地标签）；未安装时输出空串。
    "${VENV_DIR}/bin/python" -c \
        "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || true
}

TORCH_INSTALLED="$(installed_pkg_version torch)"
TORCHVISION_INSTALLED="$(installed_pkg_version torchvision)"
# 跳过条件：主版本号一致，且本地标签匹配（或安装的是 PyPI 无标签构建——
# 对应下方镜像回退路径，强行重装只会再次失败）。
torch_already_installed() {
    local installed want_base="$1"
    installed="$2"
    [[ -n "${installed}" ]] || return 1
    [[ "${installed%%+*}" == "${want_base}" ]] || return 1
    local tag=""
    [[ "${installed}" == *+* ]] && tag="${installed##*+}"
    [[ -z "${tag}" || "${tag}" == "${CUDA_TAG}" ]]
}

if torch_already_installed "${TORCH_VERSION}" "${TORCH_INSTALLED}" \
    && torch_already_installed "${TORCHVISION_VERSION}" "${TORCHVISION_INSTALLED}"; then
    log "已安装 torch ${TORCH_INSTALLED} / torchvision ${TORCHVISION_INSTALLED}，跳过安装"
else
    log "torch 现状：torch=${TORCH_INSTALLED:-未安装} torchvision=${TORCHVISION_INSTALLED:-未安装}"
    # 显式带上 +<cuda_tag> 本地版本号，强制从 PyTorch 官方索引取 CUDA 构建，
    # 避免解析到 PyPI 普通 2.10.0（其 nvidia-* 依赖固定为 cu12 系列，可能与驱动不匹配）。
    if [[ "${CUDA_TAG}" == "cpu" ]]; then
        TORCH_SPEC="torch==${TORCH_VERSION}+cpu"
        TORCHVISION_SPEC="torchvision==${TORCHVISION_VERSION}+cpu"
    else
        TORCH_SPEC="torch==${TORCH_VERSION}+${CUDA_TAG}"
        TORCHVISION_SPEC="torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}"
    fi

    log "安装 ${TORCH_SPEC} / ${TORCHVISION_SPEC}"
    PYTORCH_ARGS=(--index "${PYTORCH_INDEX_BASE}/${CUDA_TAG}")
    [[ -n "${PYPI_INDEX}" ]] && PYTORCH_ARGS+=(--index "${PYPI_INDEX}")
    if ! uv pip install --python "${VENV_DIR}/bin/python" \
        "${PYTORCH_ARGS[@]}" \
        "${TORCH_SPEC}" "${TORCHVISION_SPEC}"; then
        if [[ "${USE_MIRROR}" != "1" ]]; then
            log "错误：PyTorch 索引安装失败（网络或代理问题？）。国内网络可尝试：bash install.sh --use-mirror"
            exit 1
        fi
        # 镜像模式下回退：从 PyPI 镜像装普通版 torch（cu12 构建，与当前驱动向后兼容）
        log "警告：PyTorch wheel 索引（${PYTORCH_INDEX_BASE}/${CUDA_TAG}）不可达，"
        log "       回退：从 PyPI 安装 torch==${TORCH_VERSION}（cu12 构建，与当前驱动向后兼容）"
        uv pip install --python "${VENV_DIR}/bin/python" \
            "${DEP_ARGS[@]}" \
            "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"
    fi
fi

# --- 4. 其余依赖：以 pyproject.toml 为唯一来源 ----------------------------------
log "安装通用依赖（来源：pyproject.toml${PYPI_INDEX:+，索引 ${PYPI_INDEX}}）"
uv pip install --python "${VENV_DIR}/bin/python" \
    "${DEP_ARGS[@]}" \
    -r "${REPO_ROOT}/pyproject.toml"

# --- 5. xensesdk（私有可选包，uv 失败则退回 pip） --------------------------------
if [[ "${SKIP_XENSESDK:-0}" != "1" ]]; then
    log "尝试安装 xensesdk（私有包，可能不在公共索引）"
    if uv pip install --python "${VENV_DIR}/bin/python" "${DEP_ARGS[@]}" xensesdk; then
        log "xensesdk 安装成功（uv）"
    else
        log "uv 安装 xensesdk 失败，退回 pip 安装"
        "${VENV_DIR}/bin/python" -m ensurepip --upgrade >/dev/null 2>&1 || true
        if "${VENV_DIR}/bin/python" -m pip install "${DEP_ARGS[@]}" xensesdk; then
            log "xensesdk 安装成功（pip）"
        else
            log "警告：xensesdk 安装失败。仅 scripts/build_labeled_dataset.py 的 annotate 阶段需要，可稍后手动安装。"
        fi
    fi
fi

log "完成。运行示例："
log "  ${VENV_DIR}/bin/python scripts/train_multitask.py --help"
log "  uv run python scripts/train_multitask.py --config configs/multitask/vit_base_patch16_dinov3_lvd1689m.yaml"
