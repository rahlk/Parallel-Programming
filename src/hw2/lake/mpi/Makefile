gpu: lake_gpu.cu
	@- nvcc -c lake_gpu.cu -Xcompiler -O2  -I/usr/mpi/gcc/openmpi-1.5.4/include
	@- mpic++ lake_gpu.o -o lake -lm -L/usr/local/cuda/lib64/ -lcudart
	@- mpirun -np 5 lake

cpu: lake.c
	@- mpicc -o lake lake.c
	@- mpirun -np 5 lake

clean:
	rm -f lake
