build: imgproc_cy.pyx
	cython -3 -a imgproc_cy.pyx
	gcc -shared -pthread -fPIC $$(python3.5-config --cflags) -fno-strict-aliasing -o imgproc_cy.so imgproc_cy.c

init: ugly.pyx
	cython -3 -a ugly.pyx
	gcc -shared -pthread -fPIC $$(python3.5-config --cflags) -fno-strict-aliasing -o ugly.so ugly.c
