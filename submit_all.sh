#!/bin/bash
# =============================================================================
# 智能提交脚本 — 读取 train_configs.yaml, 按 use_gpu 和 hpo 分组提交 SLURM 任务
# 或直接在本地调试运行
# =============================================================================
# 用法:
#   ./submit_all.sh                          # SLURM 模式 (默认)
#   TARGET=local ./submit_all.sh             # 本地调试模式 (直接运行 train_runner.py)
#   TARGET=slurm ./submit_all.sh             # 显式指定 SLURM 模式
#   MODE=standalone ./submit_all.sh          # standalone 模式 (仅 SLURM)
#   CONFIG_FILE=my_configs.yaml ./submit_all.sh  # 自定义配置文件
#   DRY_RUN=1 ./submit_all.sh               # 仅打印, 不实际执行
#
#   本地调试模式可选: 只跑指定 ID
#   TARGET=local ID=3 ./submit_all.sh        # 本地只跑实验 3
#   TARGET=local ID="1 3 5" ./submit_all.sh  # 本地跑实验 1,3,5
# =============================================================================
#
# 分组逻辑 (仅 SLURM 模式):
#   Group 1: GPU + Train   — commoncpu,commongpu + GPU, 标准时限
#   Group 2: GPU + HPO     — commoncpu,commongpu + GPU, 更长时限
#   Group 3: CPU + Train   — commoncpu,new_conf, 标准时限
#   Group 4: CPU + HPO     — commoncpu,new_conf, 更长时限
# =============================================================================

set -euo pipefail

# =============================================================================
# 可调整的全局资源配置变量
# 可通过环境变量覆盖, 例如: GPU_CPUS=16 ./submit_all.sh
# =============================================================================

# GPU 任务资源配置 (use_gpu=true 的实验)
GPU_PARTITION="${GPU_PARTITION:-commoncpu,commongpu}"
GPU_GPUS="${GPU_GPUS:-1}"                       # GPU 数量, 如 "1" 或 "2"
GPU_CPUS="${GPU_CPUS:-8}"                       # CPU 核心数
GPU_MEM="${GPU_MEM:-100G}"                      # 内存
GPU_THREADS_PER_CORE=""                         # 超线程设置, 空表示不设置 (保持默认)
GPU_TRAIN_TIME="${GPU_TRAIN_TIME:-1-0:0:0}"     # GPU 训练时限
GPU_HPO_TIME="${GPU_HPO_TIME:-7-0:0:0}"         # GPU HPO 时限

# CPU 任务资源配置 (use_gpu=false 的实验)
CPU_PARTITION="${CPU_PARTITION:-commoncpu,new_conf}"
CPU_CPUS="${CPU_CPUS:-100}"                     # CPU 核心数
CPU_THREADS_PER_CORE="${CPU_THREADS_PER_CORE:-1}" # 超线程设置, "1"=关闭超线程
CPU_MEM="${CPU_MEM:-500G}"                      # 内存
CPU_TRAIN_TIME="${CPU_TRAIN_TIME:-1-0:0:0}"     # CPU 训练时限
CPU_HPO_TIME="${CPU_HPO_TIME:-7-0:0:0}"         # CPU HPO 时限
# =============================================================================

TARGET=${TARGET:-slurm}
CONFIG_FILE=${CONFIG_FILE:-train_configs.yaml}
MODE=${MODE:-unified}
DRY_RUN=${DRY_RUN:-0}
ID=${ID:-all}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_SCRIPT="submit.sh"

cd "${SCRIPT_DIR}"

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: config file '${CONFIG_FILE}' not found"
    exit 1
fi

if [ ! -f "${SUBMIT_SCRIPT}" ] && [ "${TARGET}" != "local" ]; then
    echo "Error: submit.sh not found at '${SUBMIT_SCRIPT}'"
    exit 1
fi

# 用 Python 解析 YAML, 输出 "id use_gpu" 对 + 全局 hpo_enabled
parse_result=$(python3 -c "
import yaml, sys
with open('${CONFIG_FILE}', 'r') as f:
    cfg = yaml.safe_load(f)
hpo_enabled = cfg.get('shared_hpo', {}).get('enabled', False)
experiments = cfg.get('experiments', [])
for i, exp in enumerate(experiments, 1):
    use_gpu = exp.get('use_gpu', False)
    print(f'{i} {use_gpu} {hpo_enabled}')
" 2>&1)

if [ $? -ne 0 ]; then
    echo "Error parsing YAML: ${parse_result}"
    exit 1
fi

# 分组
gpu_train_ids=()
gpu_hpo_ids=()
cpu_train_ids=()
cpu_hpo_ids=()
all_ids=()

while read -r id use_gpu hpo_enabled; do
    all_ids+=("${id}")
    if [ "${use_gpu}" = "True" ] && [ "${hpo_enabled}" = "True" ]; then
        gpu_hpo_ids+=("${id}")
    elif [ "${use_gpu}" = "True" ]; then
        gpu_train_ids+=("${id}")
    elif [ "${hpo_enabled}" = "True" ]; then
        cpu_hpo_ids+=("${id}")
    else
        cpu_train_ids+=("${id}")
    fi
done <<< "${parse_result}"

echo "============================================"
echo "Config: ${CONFIG_FILE}"
echo "Target: ${TARGET}"
echo "Mode:   ${MODE}"
echo "============================================"
echo ""

# --- 本地调试模式 ---
if [ "${TARGET}" = "local" ]; then
    # 获取 Python 路径
    PYTHON_PATH=${PYTHON_PATH:-python}

    if [ "${ID}" = "all" ]; then
        run_ids=("${all_ids[@]}")
    else
        read -ra run_ids <<< "${ID}"
    fi

    echo "Local debug mode:"
    echo "  Experiments: ${run_ids[*]}"
    echo ""

    if [ "${DRY_RUN}" = "1" ]; then
        for rid in "${run_ids[@]}"; do
            echo "  [DRY RUN] ${PYTHON_PATH} -u train_runner.py --config ${CONFIG_FILE} --id ${rid}"
        done
        echo ""
        echo "============================================"
        echo "Dry run complete."
        echo "============================================"
        exit 0
    fi

    for rid in "${run_ids[@]}"; do
        echo "--------------------------------------------"
        echo "Running experiment ${rid} locally..."
        echo "--------------------------------------------"
        ${PYTHON_PATH} -u train_runner.py --config ${CONFIG_FILE} --id ${rid}
        echo ""
    done

    echo "============================================"
    echo "All local runs completed."
    echo "============================================"
    exit 0
fi

# --- SLURM 模式 ---
echo "Group summary:"
echo "  GPU + Train: ${gpu_train_ids[*]:-none}"
echo "  GPU + HPO:   ${gpu_hpo_ids[*]:-none}"
echo "  CPU + Train: ${cpu_train_ids[*]:-none}"
echo "  CPU + HPO:   ${cpu_hpo_ids[*]:-none}"
echo ""

ids_to_array_str() {
    local ids=("$@")
    if [ ${#ids[@]} -eq 0 ]; then
        echo ""
        return
    fi
    local result="${ids[0]}"
    for i in $(seq 1 $((${#ids[@]} - 1))); do
        result="${result},${ids[$i]}"
    done
    echo "${result}"
}

submit_job() {
    local partition="$1"
    local gres_flag="$2"
    local time_limit="$3"
    local cpus_per_task="$4"
    local threads_per_core="$5"
    local mem="$6"
    local array_str="$7"
    local label="$8"

    if [ -z "${array_str}" ]; then
        return
    fi

    echo "--------------------------------------------"
    echo "Group: ${label}"
    echo "  Partition:       ${partition}"
    echo "  GPU:             ${gres_flag:-none}"
    echo "  Time:            ${time_limit}"
    echo "  CPUs per task:   ${cpus_per_task}"
    echo "  Threads/core:    ${threads_per_core:-default}"
    echo "  Memory:          ${mem}"
    echo "  Array:           ${array_str}"
    echo "  Mode:            ${MODE}"
    echo "--------------------------------------------"

    if [ "${DRY_RUN}" = "1" ]; then
        local dry_cmd="sudo -u hpc_user_zhangjunjie sbatch --partition=${partition}"
        if [ -n "${gres_flag}" ]; then
            dry_cmd="${dry_cmd} ${gres_flag}"
        fi
        dry_cmd="${dry_cmd} --time=${time_limit} --cpus-per-task=${cpus_per_task} --mem=${mem}"
        if [ -n "${threads_per_core}" ]; then
            dry_cmd="${dry_cmd} --threads-per-core=${threads_per_core}"
        fi
        dry_cmd="${dry_cmd} --array=${array_str} --export=MODE=${MODE},CONFIG_FILE=${CONFIG_FILE} ${SUBMIT_SCRIPT}"
        echo "  [DRY RUN] ${dry_cmd}"
        echo ""
        return
    fi

    local sbatch_cmd="sbatch --partition=${partition}"
    if [ -n "${gres_flag}" ]; then
        sbatch_cmd="${sbatch_cmd} ${gres_flag}"
    fi
    sbatch_cmd="${sbatch_cmd} --time=${time_limit}"
    sbatch_cmd="${sbatch_cmd} --cpus-per-task=${cpus_per_task}"
    sbatch_cmd="${sbatch_cmd} --mem=${mem}"
    if [ -n "${threads_per_core}" ]; then
        sbatch_cmd="${sbatch_cmd} --threads-per-core=${threads_per_core}"
    fi
    sbatch_cmd="${sbatch_cmd} --array=${array_str}"
    sbatch_cmd="${sbatch_cmd} --export=MODE=${MODE},CONFIG_FILE=${CONFIG_FILE}"
    sbatch_cmd="${sbatch_cmd} ${SUBMIT_SCRIPT}"

    echo "  Submitting: sudo -u hpc_user_zhangjunjie ${sbatch_cmd}"
    output=$(sudo -u hpc_user_zhangjunjie ${sbatch_cmd} 2>&1) || true
    if echo "${output}" | grep -qi "submitted"; then
        echo "  => ${output}"
    else
        echo "  => FAILED: ${output}"
    fi
    echo ""
}

# Group 1: GPU + Train
if [ ${#gpu_train_ids[@]} -gt 0 ]; then
    gpu_train_array=$(ids_to_array_str "${gpu_train_ids[@]}")
    if [ -n "${GPU_GPUS}" ]; then gpu_gres="--gres=gpu:${GPU_GPUS}"; else gpu_gres=""; fi
    submit_job "${GPU_PARTITION}" "${gpu_gres}" "${GPU_TRAIN_TIME}" "${GPU_CPUS}" "${GPU_THREADS_PER_CORE}" "${GPU_MEM}" "${gpu_train_array}" "GPU + Train"
fi

# Group 2: GPU + HPO (更长时限)
if [ ${#gpu_hpo_ids[@]} -gt 0 ]; then
    gpu_hpo_array=$(ids_to_array_str "${gpu_hpo_ids[@]}")
    if [ -n "${GPU_GPUS}" ]; then gpu_gres="--gres=gpu:${GPU_GPUS}"; else gpu_gres=""; fi
    submit_job "${GPU_PARTITION}" "${gpu_gres}" "${GPU_HPO_TIME}" "${GPU_CPUS}" "${GPU_THREADS_PER_CORE}" "${GPU_MEM}" "${gpu_hpo_array}" "GPU + HPO"
fi

# Group 3: CPU + Train
if [ ${#cpu_train_ids[@]} -gt 0 ]; then
    cpu_train_array=$(ids_to_array_str "${cpu_train_ids[@]}")
    submit_job "${CPU_PARTITION}" "" "${CPU_TRAIN_TIME}" "${CPU_CPUS}" "${CPU_THREADS_PER_CORE}" "${CPU_MEM}" "${cpu_train_array}" "CPU + Train"
fi

# Group 4: CPU + HPO (更长时限)
if [ ${#cpu_hpo_ids[@]} -gt 0 ]; then
    cpu_hpo_array=$(ids_to_array_str "${cpu_hpo_ids[@]}")
    submit_job "${CPU_PARTITION}" "" "${CPU_HPO_TIME}" "${CPU_CPUS}" "${CPU_THREADS_PER_CORE}" "${CPU_MEM}" "${cpu_hpo_array}" "CPU + HPO"
fi

echo "============================================"
echo "All jobs submitted."
echo "============================================"
