#!/usr/bin/env bash
# 顺序运行 scripts/train_with_timm.py，依次喂入 configs/*.yaml。
#
# 用法：
#   ./scripts/run_all_timm.sh                       # 跑 configs/*.yaml 全部 6 个
#   ./scripts/run_all_timm.sh resnet50_a1_in1k      # 只跑指定 config（可省略 .yaml）
#   ./scripts/run_all_timm.sh resnet50 convnextv2_base_fcmae_ft_in22k_in1k
#
# 单个 config 失败不会中断后续 config；最后会打印汇总。
# 每次运行的 stdout/stderr 写入 logs/<config>_<timestamp>.log，
# 同时通过 tee 实时回显到终端。

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="configs"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

if [[ $# -gt 0 ]]; then
    REQUESTED=("$@")
else
    REQUESTED=()
    while IFS= read -r -d '' f; do
        REQUESTED+=("$(basename "$f")")
    done < <(find "$CONFIG_DIR" -maxdepth 1 -name '*.yaml' -print0 | sort -z)
fi

if [[ ${#REQUESTED[@]} -eq 0 ]]; then
    echo "No configs found under $CONFIG_DIR." >&2
    exit 1
fi

declare -a CONFIGS=()
for name in "${REQUESTED[@]}"; do
    [[ "$name" == *.yaml ]] || name="${name}.yaml"
    path="$CONFIG_DIR/$name"
    if [[ ! -f "$path" ]]; then
        echo "Config not found: $path" >&2
        exit 1
    fi
    CONFIGS+=("$path")
done

echo "Will run ${#CONFIGS[@]} config(s):"
for c in "${CONFIGS[@]}"; do
    echo "  - $c"
done
echo

declare -a OK=()
declare -a FAIL=()

for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .yaml)"
    ts="$(date +%Y%m%d-%H%M%S)"
    log_path="$LOG_DIR/${stem}_${ts}.log"

    echo "=============================================================="
    echo "[ $(date '+%F %T') ] START  $cfg"
    echo "log -> $log_path"
    echo "=============================================================="

    start=$SECONDS
    if python scripts/train_with_timm.py --config "$cfg" 2>&1 | tee "$log_path"; then
        elapsed=$((SECONDS - start))
        printf '[ %s ] OK     %s  (%dm%ds)\n' "$(date '+%F %T')" "$cfg" $((elapsed/60)) $((elapsed%60))
        OK+=("$cfg")
    else
        rc=${PIPESTATUS[0]}
        elapsed=$((SECONDS - start))
        printf '[ %s ] FAIL   %s  rc=%d (%dm%ds)\n' "$(date '+%F %T')" "$cfg" "$rc" $((elapsed/60)) $((elapsed%60))
        FAIL+=("$cfg (rc=$rc)")
    fi
    echo
done

echo "=============================================================="
echo "Summary: ${#OK[@]} ok, ${#FAIL[@]} failed"
for c in "${OK[@]}";   do echo "  OK    $c"; done
for c in "${FAIL[@]}"; do echo "  FAIL  $c"; done
echo "=============================================================="

[[ ${#FAIL[@]} -eq 0 ]]
