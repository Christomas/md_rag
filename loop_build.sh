#!/bin/bash
# loop_build.sh: 自动重启构建任务，防止内存泄漏

export PYTHONPATH=.

# 配置参数
BATCH_SIZE=1      # 降回 1，保证桌面操作流畅
FETCH_SIZE=64     # 适中的数据库读取批次
LIMIT_PER_RUN=5000 # 延长单次运行时间，每 5000 条重启一次
DEVICE="mps"       # 依然使用 GPU，但在 batch=1 下压力较小

echo "=== 开始循环构建任务 ==="
echo "策略: 每处理 $LIMIT_PER_RUN 条记录重启一次进程以重置显存"

while true; do
    # 运行构建命令，并同时将输出打印到屏幕和临时文件
    python3 src/builder.py --device $DEVICE --batch_size $BATCH_SIZE --fetch_size $FETCH_SIZE --limit $LIMIT_PER_RUN | tee build.log
    
    # 获取退出码
    EXIT_CODE=${PIPESTATUS[0]} # 获取管道前一个命令（python）的退出码
    
    # 检查是否完成
    if grep -q "所有切片均已有向量" build.log; then
        echo "=== 🎉 全部任务完成！自动退出循环。 ==="
        rm build.log
        break
    fi
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "检测到错误 (Exit Code: $EXIT_CODE)，3秒后尝试重启..."
        sleep 3
    else
        echo "本轮任务完成，正在重启进程清理内存..."
        sleep 1
    fi
done
