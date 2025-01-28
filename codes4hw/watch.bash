#!/bin/bash  

# 要检查的进程 ID  
PID=$1  

# 要执行的其他命令  
OTHER_COMMAND="echo 'The process is not running. Executing other command.'"  

# 检查 PID 是否在运行  
while true; do  
    if ps -p $PID > /dev/null; then  
        echo "Process $PID is running. Checking again in 30 seconds..."  
        sleep 30  
    else  
        echo "Process $PID is not running. Executing other command."  
        eval $OTHER_COMMAND  
        break  
    fi  
done
