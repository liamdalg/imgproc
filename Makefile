build: convolution.pyx
	cython -3 -a convolution.pyx
	gcc -shared -pthread -fPIC $$(python3-config --cflags) -fno-strict-aliasing -o convolution.so convolution.c
