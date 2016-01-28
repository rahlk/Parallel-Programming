nodes8:
	@- mpicc -o p1 p1.c
	@- echo "Running on 8 nodes.."
	@- mpirun -np 8 p1 > nodes8.dat
	@- echo "Running on 16 nodes.."
	@- mpirun -np 16 p1 > nodes16.dat

# nodes16:
# 	@- mpicc -o p1 p1.c
