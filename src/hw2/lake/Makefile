all: lake

lake: lakegpu_2.cu lake.cu
	nvcc lakegpu_2.cu lake.cu -o ./lake -Xcompiler -O2 -lm

clean: 
	rm -f lake
