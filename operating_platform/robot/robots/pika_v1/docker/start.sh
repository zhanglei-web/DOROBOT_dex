#!/bin/bash

# 配置参数
CONTAINER_NAME=dr-container-pika-v1
PROJECT_DIR="/Operating-Platform"
CONDA_ENV1="op"
CONDA_ENV2="op-component-vive"
CONDA_ENV3="op-robot-pika"
DATAFLOW_PATH_COMPONENT_VIVE="operating_platform/robot/components/tracker_6d_vive/dora_vive_dual.yml"
DATAFLOW_PATH_COMPONENT_GRIPPER="operating_platform/robot/components/gripper_pika/dora_gripper_pika_dual.yml"
DATAFLOW_PATH_ROBOT_PIKA="operating_platform/robot/robots/pika_v1/robot_dataflow.yml"
TIMEOUT=30  # 等待超时时间（秒）
CLEANED_UP=false
IN_CONTAINER_MODE=false  # 容器模式标志
OVERWRITE_LOGS=false     # 是否覆盖日志标志

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo
    echo "选项:"
    echo "  -c, --container      在容器内直接运行（容器模式）"
    echo "  -h, --host           在宿主机运行（控制容器） - 默认模式"
    echo "  -o, --overwrite-logs 覆盖过去的日志文件（清空旧日志）"
    echo "  -?, --help           显示此帮助信息"
    echo
    echo "示例:"
    echo "  $0                     # 宿主机模式（默认），追加日志"
    echo "  $0 --container         # 容器模式，追加日志"
    echo "  $0 -o                  # 覆盖日志模式"
    echo "  $0 -c -o               # 容器模式 + 覆盖日志"
    exit 0
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--container)
            IN_CONTAINER_MODE=true
            shift
            ;;
        -h|--host)
            IN_CONTAINER_MODE=false
            shift
            ;;
        -o|--overwrite-logs)
            OVERWRITE_LOGS=true
            shift
            ;;
        -\?|--help)
            show_help
            ;;
        *)
            echo "错误: 未知参数 '$1'" >&2
            echo "使用 '$0 --help' 查看帮助" >&2
            exit 1
            ;;
    esac
done

# 颜色定义
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] \033[0;32m$1\033[0m"
}

error() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] \033[0;31m错误: $1\033[0m" >&2
    exit 1
}

# 清理函数
cleanup() {
    if $CLEANED_UP; then
        log "清理已执行过，跳过重复清理"
        return
    fi
    CLEANED_UP=true

    log "正在清理..."
    # 终止本地记录的进程
    if [ -f ".pids" ]; then
        kill $(cat .pids 2>/dev/null) 2>/dev/null || true
        rm -f .pids
    fi

    # 容器模式下跳过容器相关操作
    if [ "$IN_CONTAINER_MODE" = false ]; then
        # 终止容器内的进程（如果容器存在）
        if docker inspect --type=container "$CONTAINER_NAME" &>/dev/null; then
            log "尝试终止容器内的关键进程..."

            # 定义一个辅助函数来执行命令并输出结果
            execute_in_container_with_result() {
                local cmd="$1"
                local desc="$2"
                log "执行容器内命令: $desc"
                local result=$(docker exec "$CONTAINER_NAME" sh -c "$cmd" 2>&1)
                if [ $? -eq 0 ]; then
                    log "$desc 成功：$result"
                else
                    log "$desc 失败或未找到匹配进程（可能已退出）"
                fi
            }

            # 终止容器内相关进程
            execute_in_container_with_result "pkill -f '$DATAFLOW_PATH_COMPONENT_VIVE'" "终止 dora VIVE 数据流进程"
            execute_in_container_with_result "pkill -f '$DATAFLOW_PATH_COMPONENT_GRIPPER'" "终止 dora GRIPPER 数据流进程"
            execute_in_container_with_result "pkill -f '$DATAFLOW_PATH_ROBOT_PIKA'" "终止 dora ROBOT 数据流进程"
            execute_in_container_with_result "pkill -f 'coordinator.py'" "终止 coordinator.py 进程"
            execute_in_container_with_result "pkill -f 'dora-daemon'" "终止 dora-daemon 守护进程"

            # 可选：等待几秒确保进程优雅退出
            sleep 2
        fi
        
        # 停止容器（如果仍在运行）
        if docker inspect --type=container "$CONTAINER_NAME" &>/dev/null; then
            log "停止容器 $CONTAINER_NAME..."
            docker stop "$CONTAINER_NAME" > /dev/null 2>&1 || true
        fi
        
        # 恢复X11访问权限
        xhost - >/dev/null 2>&1 || true
    fi
}

# 捕获中断信号
trap cleanup INT TERM EXIT

# 检查容器是否存在
check_container() {
    if ! docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        error "容器 $CONTAINER_NAME 不存在，请先创建容器"
    fi
}

# 等待容器就绪（检查视频设备）
wait_for_container() {
    log "等待容器就绪..."
    local start_time=$(date +%s)
    
    while ! docker exec "$CONTAINER_NAME" sh -c "ls /dev/video* >/dev/null 2>&1"; do
        sleep 1
        local current_time=$(date +%s)
        if (( current_time - start_time > TIMEOUT )); then
            error "等待视频设备挂载超时"
        fi
    done
    log "视频设备已就绪"
}

# 执行容器内命令 (宿主机模式)
execute_in_container() {
    local cmd="$1"
    local log_file="$2"
    
    log "执行命令: $cmd"
    docker exec -t "$CONTAINER_NAME" bash -c "$cmd" \
        > >(tee -a "$log_file") 2>&1 &
    echo $! >> .pids
}

# 执行本地命令 (容器模式)
execute_local() {
    local cmd="$1"
    local log_file="$2"
    
    log "在容器内执行命令: $cmd"
    (eval "$cmd") > >(tee -a "$log_file") 2>&1 &
    echo $! >> .pids
}

# 主程序
if [ "$IN_CONTAINER_MODE" = false ]; then
    log "运行模式: 宿主机模式（控制容器）"
    xhost + >/dev/null 2>&1 || error "无法设置 X11 转发"

    log "停止旧容器..."
    docker stop "$CONTAINER_NAME" >/dev/null 2>&1 || true

    log "启动容器..."
    docker start "$CONTAINER_NAME" || error "容器启动失败"

    check_container
    wait_for_container
else
    log "运行模式: 容器模式（直接在容器内运行）"
    # 容器模式下确保日志目录存在
    mkdir -p logs
fi

# 确保日志目录存在（两种模式都需要）
mkdir -p logs

# 处理日志覆盖选项
if [ "$OVERWRITE_LOGS" = true ]; then
    log "覆盖日志模式：清空旧日志文件"
    > logs/coordinator.log
    > logs/dataflow_vive.log
    > logs/dataflow_gripper.log
    > logs/dataflow_pika.log
else
    log "追加日志模式：将新日志追加到现有文件"
fi

# 准备conda激活命令
CONDA_ACTIVATE="source /opt/conda/etc/profile.d/conda.sh && conda activate"

# 清理旧的PID文件
rm -f .pids

# 根据运行模式选择执行函数
if [ "$IN_CONTAINER_MODE" = false ]; then
    EXEC_FUNCTION="execute_in_container"
else
    EXEC_FUNCTION="execute_local"
fi

log "启动协调器..."
$EXEC_FUNCTION "cd $PROJECT_DIR && $CONDA_ACTIVATE $CONDA_ENV1 && python operating_platform/core/coordinator.py --robot.type=pika_v1" "logs/coordinator.log"

sleep 3 

# 并行执行任务
log "启动VIVE数据流..."
$EXEC_FUNCTION "cd $PROJECT_DIR && $CONDA_ACTIVATE $CONDA_ENV2 && dora run $DATAFLOW_PATH_COMPONENT_VIVE" "logs/dataflow_vive.log"

log "启动GRIPPER数据流..."
$EXEC_FUNCTION "cd $PROJECT_DIR && $CONDA_ACTIVATE $CONDA_ENV2 && dora run $DATAFLOW_PATH_COMPONENT_GRIPPER" "logs/dataflow_gripper.log"

log "启动ROBOT数据流..."
$EXEC_FUNCTION "cd $PROJECT_DIR && $CONDA_ACTIVATE $CONDA_ENV3 && dora run $DATAFLOW_PATH_ROBOT_PIKA" "logs/dataflow_pika.log"

log "所有进程已启动，PID记录在.pids文件中"
log "监控日志文件："
echo "- logs/coordinator.log"
echo "- logs/dataflow_vive.log"
echo "- logs/dataflow_gripper.log"
echo "- logs/dataflow_pika.log"

# 持续运行直到收到中断信号
wait