#!/usr/bin/env bash
# 顺序运行 scripts/train_with_timm.py，跑 MAE 掩码预训练（token-drop MAE）。
# MAE 只适用于 ViT backbone，因此默认跑 configs 下的 *_mae.yaml（当前为 3 个 dinov3 ViT：small/base/large）。
#
# 用法：
#   ./scripts/run_all_mae.sh                                  # 跑全部 *_mae.yaml
#   ./scripts/run_all_mae.sh vit_base_patch16_dinov3_lvd1689m_mae   # 只跑指定（可省略 .yaml）
#   ./scripts/run_all_mae.sh vit_small_patch16_dinov3_lvd1689m_mae vit_large_patch16_dinov3_lvd1689m_mae
#
# 单个 config 失败不会中断后续；最后打印汇总。日志写入 logs/<config>_<timestamp>.log。

set -u
set -o pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG_DIR="configs"
LOG_DIR="logs"
TASK="mae"
mkdir -p "$LOG_DIR"

if [[ $# -gt 0 ]]; then
    REQUESTED=("$@")
else
    # 默认：configs 下全部 *_mae.yaml。
    REQUESTED=()
    while IFS= read -r -d '' f; do
        REQUESTED+=("$(basename "$f")")
    done < <(find "$CONFIG_DIR" -maxdepth 1 -name '*_mae.yaml' -print0 | sort -z)
fi

if [[ ${#REQUESTED[@]} -eq 0 ]]; then
    echo "No *_mae.yaml configs found under $CONFIG_DIR." >&2
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
    log_path="$LOG_DIR/${stem}_${ts}.log"

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
