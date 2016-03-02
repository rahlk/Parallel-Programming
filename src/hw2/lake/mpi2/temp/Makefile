lake:	lake_mpi.cu  lakegpu_mpi.cu
	@- nvcc -c lakegpu_mpi.cu -Xcompiler -O -I/usr/mpi/gcc/openmpi-1.5.4/include -arch=sm_20
	@- nvcc -c lake_mpi.cu -Xcompiler -O2  -I/usr/mpi/gcc/openmpi-1.5.4/include
	@- mpic++ lakegpu_mpi.o lake_mpi.o -o lake -lm -L/usr/local/cuda/lib64/ -lcudart

run: lake
	@- mpirun -np 5 lake 256 9 1 1024 1
clean:
	rm -f lake
