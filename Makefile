all:
	nvcc -shared  -o mandelbrot.so -Xcompiler -fPIC mandelbrot.cu
