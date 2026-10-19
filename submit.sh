#!/bin/bash
#SBATCH -J zjj_short_term_pred
##SBATCH -p commoncpu,commongpu       # 分区由 submit_all.sh 通过 sbatch --partition 传入
#SBATCH -t 1-0:0:0
#SBATCH --chdir /cache/hpc_user_zhangjunjie
#SBATCH --mem 500G
##SBATCH --gres=gpu:1                 # GPU 由 submit_all.sh 通过 sbatch --gres 传入
##SBATCH --cpus-per-task=100           # CPU 数由 submit_all.sh 通过 sbatch --cpus-per-task 传入
##SBATCH --threads-per-core=1          # 超线程设置由 submit_all.sh 传入
#SBATCH -o ./tmp/log/%x_%A_%a.out
#SBATCH -e ./tmp/log/%x_%A_%a.err
##SBATCH --array=1-5                  # array 范围由 submit_all.sh 通过 sbatch --array 传入

# =============================================================================
# 训练任务执行脚本 (由 submit_all.sh 调用, 不要直接 sbatch 此文件)
#
# 分区和 GPU 设置通过 sbatch 命令行参数传入:
#   GPU任务:  sbatch --partition=commoncpu,commongpu --gres=gpu:1 --cpus-per-task=8 --array=1-4 submit.sh
#   CPU任务:  sbatch --partition=commoncpu,new_conf --cpus-per-task=100 --threads-per-core=1 --array=5-5 submit.sh
# =============================================================================

MODE=${MODE:-unified}
CONFIG_FILE=${CONFIG_FILE:-train_configs.yaml}

export HADOOP_USER_NAME=zli
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
export HADOOP_HOME=/opt/hadoop
export LD_LIBRARY_PATH=${JAVA_HOME}/lib/server:$HADOOP_HOME/lib/native/:${LD_LIBRARY_PATH}
export CLASSPATH=$CLASSPATH:`$HADOOP_HOME/bin/hadoop classpath --glob`

# --- 环境准备 ---
env_name=modeltrainer
env_path=~/zjj_miniforge_array${SLURM_ARRAY_TASK_ID}
rm -rf ${env_path}
mkdir -p ${env_path}
/opt/hadoop/bin/hadoop fs -get -t 1 -f /ext1/user/zli/zjj/env_list/${env_name}.tar.gz ${env_path}
tar -zxf ${env_path}/${env_name}.tar.gz -C ${env_path}
rm -f ${env_path}/${env_name}.tar.gz
touch ${env_path}/success.txt

# --- 工作目录 ---
workdir=./zjj/${SLURM_ARRAY_TASK_ID}/zjj
mkdir -p ${workdir}
cd ${workdir}

/opt/hadoop/bin/hadoop fs -get -t 1000 -f "/ext1/user/zli/zjj/ResampleData" ./

# --- 下载项目 ---
hdfs_path=/ext1/user/zli/zjj/model_trainer
pj_name=modeltrainer
rm -rf ${pj_name}
/opt/hadoop/bin/hadoop fs -get -t 1000 -f "${hdfs_path}/${pj_name}" ./

export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
export PATH=/usr/bin:/usr/local/bin:/bin:/sbin:$PATH

# 强制 Triton/torch.compile 使用系统自带的完整 GCC，忽略打包环境里可能残缺的编译器
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

cd ${pj_name}
pwd

# 将 ./lib 换成绝对路径，避免后续找不到动态链接库
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH

PYTHON_PATH=${env_path}/${env_name}/bin/python

# --- 按模式执行 ---
if [ "$MODE" = "unified" ]; then
    echo "=========================================="
    echo "MODE: unified (train_runner.py)"
    echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
    echo "Config: ${CONFIG_FILE}"
    echo "=========================================="
    "${PYTHON_PATH}" -u train_runner.py --config ${CONFIG_FILE} --id ${SLURM_ARRAY_TASK_ID}
else
    echo "=========================================="
    echo "MODE: standalone (generate_runners.py)"
    echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
    echo "Config: ${CONFIG_FILE}"
    echo "=========================================="
    "${PYTHON_PATH}" -u generate_runners.py --config ${CONFIG_FILE}
    py_name=$(printf "python_train_runner_%06d.py" "${SLURM_ARRAY_TASK_ID}")
    echo "Running: ${py_name}"
    "${PYTHON_PATH}" -u ${py_name}
fi
