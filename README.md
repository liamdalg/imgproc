# very WIP imgproc

An application I'm building in Python/Cython with Numpy to process images. VERY rough around the edges but try it out with:

* `$ make`
    * If make fails then change `python3-config` to your relevant version of python
* Open Python
* `import imgproc` and `import cv2`
* `img = cv2.imread('<file>', 0)`
* `out = imgproc.sobel_operator(img, 70, True)`
* `cv2.imwrite('<outfile>', out)`

<div align="center">
    <img style="width: 40%" src="examples/circle.jpg"/>
    <img style="width: 40%" src="examples/circle-grad.jpg"/>
</div>

<div align="center">
    <img style="width: 40%" src="examples/valve.png"/>
    <img style="width: 40%" src="examples/valve-grad.png"/> <br>
Right now the gradient colouring is very susceptible to noise, sorry.
</div>

## Todo

- [x] Cython kernel convolutions
- [x] Gradient colouring in sobel operator
- [ ] Add fused types
- [ ] Remove noise on sobel gradient
- [ ] Add Laplacian filter
- [ ] Implement more in Cython