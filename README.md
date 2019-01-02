# imgproc

Small python app to perform image processing such as edge detection, sharpening etc. The main part is made using Cython so it requires some tinkering to get going first.

## Usage

* Install `cv2` and `numpy`
* Run `$ make` (if this doesn't work, then change `python3-config --cflags` to the relevant version of python)
* Run `$ python -m unittest discover`
* Yay