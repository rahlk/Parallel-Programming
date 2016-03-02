build: setup.c sock_msg.c
	@- cat \$PBS_NODEFILE > NODES
	@- killall -9 setup
	@- gcc -o setup setup.c sock_msg.c
run: setup
	@- port=($(netstat -ltn | sed -rne '/^tcp/{/:(22|25)\>/d;s/.*:([0-9]+)\>.*/\1/p}')) && echo ${port[@]} | tr " " "\n" > PORTS
	@- setup 0 &
	@- echo "Server Running on " && echo host localhost
clean:
	@- rm setup
