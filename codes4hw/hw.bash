#!/bin/bash  

# 进程名称或PID  
PROCESS_NAME="your_process_name_or_pid"  

# 检查进程是否存在  
while pgrep -x "$PROCESS_NAME" > /dev/null; do  
    echo "进程 $PROCESS_NAME 正在运行，等待 30 秒..."  
    sleep 30  
done  

# 进程结束后的命令  
echo "进程 $PROCESS_NAME 已结束，执行后续命令..."  
# 在这里添加你的后续命令