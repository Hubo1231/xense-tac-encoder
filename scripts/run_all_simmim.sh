#!/usr/bin/env bash
# 顺序运行 scripts/train_with_timm.py，对每个 config 强制以 SimMIM 掩码预训练（--task simmim）。
# SimMIM 对任意 backbone 通用（卷积 / 混合 / ViT 都行），因此默认跑「除 *_mae.yaml 之外」的全部 config。
#
# 用法：
#   ./scripts/run_all_simmim.sh                       # 跑全部 backbone 的 SimMIM
#   ./scripts/run_all_simmim.sh resnet50_a1_in1k      # 只跑指定 config（可省略 .yaml）
#   ./scripts/run_all_simmim.sh resnet50 fastvit_t12_apple_dist_in1k
#
# 单个 config 失败不会中断后续；最后打印汇总。日志写入 logs/<config>_simmim_<timestamp>.log。

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="configs"
LOG_DIR="logs"
TASK="simmim"
mkdir -p "$LOG_DIR"

if [[ $# -gt 0 ]]; then
    REQUESTED=("$@")
else
    # 默认：configs 下全部 *.yaml，但排除 *_mae.yaml（那是 MAE 专用、与基础 config 同模型）。
    REQUESTED=()
    while IFS= read -r -d '' f; do
        base="$(basename "$f")"
        [[ "$base" == *_mae.yaml ]] && continue
        REQUESTED+=("$base")
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

echo "Will run ${#CONFIGS[@]} config(s) as task=$TASK:"
for c in "${CONFIGS[@]}"; do
    echo "  - $c"
done
echo

declare -a OK=()
declare -a FAIL=()

for cfg in "${CONFIGS[@]}"; do
    stem="$(basename "$cfg" .yaml)"
    ts="$(date +%Y%m%d-%H%M%S)"
    log_path="$LOG_DIR/${stem}_${TASK}_${ts}.log"

    echo "=============================================================="
    echo "[ $(date '+%F %T') ] START  $cfg  (task=$TASK)"
    echo "log -> $log_path"
    echo "=============================================================="

    start=$SECONDS
    if python scripts/train_with_timm.py --config "$cfg" --task "$TASK" 2>&1 | tee "$log_path"; then
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
echo "Summary (task=$TASK): ${#OK[@]} ok, ${#FAIL[@]} failed"
for c in "${OK[@]}";   do echo "  OK    $c"; done
for c in "${FAIL[@]}"; do echo "  FAIL  $c"; done
echo "=============================================================="

[[ ${#FAIL[@]} -eq 0 ]]
