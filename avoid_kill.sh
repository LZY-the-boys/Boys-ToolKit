is_port_in_use() {
    local port=$1
    if lsof -i:$port > /dev/null; then
        return 0
    else
        return 1
    fi
}
find_available_port() {
    while true; do
        port=$(( ( RANDOM % 10001 ) + 20000 ))
        if ! is_port_in_use $port; then
            echo $port
            return
        fi
    done
}
while true; do
    if [ -f "gpu_avoid_kill.py" ]; then
        echo "lzy: gpu_avoid_kill.py script not found. exit."
	else
		exit 1
    fi

    port=$(find_available_port)
    echo "lzy: accelerate use port: $port"
    accelerate launch --main_process_port $port gpu_avoid_kill.py &
    
    sleep 50

    if pgrep -f "gpu_avoid_kill.py" > /dev/null; then
        echo "Accelerate launched successfully."
        break
    else
        echo "Accelerate launch failed. Retrying..."
    fi
done