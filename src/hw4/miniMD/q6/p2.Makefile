configa:
	@- export OMP_NUM_THREADS=4
	@- make openmpi OMP=yes

configb:
	@- make openmpi OMP=no

