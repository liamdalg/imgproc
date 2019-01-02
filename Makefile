build: imgproc_cy.pyx
	cython -3 -a imgproc_cy.pyx
	gcc -shared -pthread -fPIC $$(python3-config --cflags) -fno-strict-aliasing -o imgproc_cy.so imgproc_cy.c
