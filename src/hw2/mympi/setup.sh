#! bin/bash
rm setup
cat $PBS_NODEFILE > NODES
killall -9 setup
gcc -o setup setup.c sock_msg.c
setup 0 3001 &
echo "Server Running on " && host localhost
