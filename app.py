import argparse
import imgproc
import cv2
from pathlib import PurePath

# TODO: add conditional argparsing, to provide variable kwargs for each operation rather than these
# default ones.
operations = {
    'sobel': lambda img: imgproc.sobel_operator(img, 70, False),
    'sobel_gradient': lambda img: imgproc.sobel_operator(img, 70, True),
    'gaussian': lambda img: imgproc.gaussian_blur(img)
}


def main():
    parser = argparse.ArgumentParser(description='Basic image processing application with kernel '
                                                 'convolution.')
    parser.add_argument('operation', choices=['sobel', 'sobel_gradient', 'gaussian', 'examples'],
                        help='The kernel operation to perform.')
    parser.add_argument('src', help='Source image path.')
    parser.add_argument('-o', help='Output image path.')
    args = parser.parse_args()

    in_path = PurePath(args.src)
    if args.o is None:
        out_path = in_path.with_name(f'{in_path.stem}_out{in_path.suffix}')
    else:
        out_path = PurePath(args.o)

    img = cv2.imread(str(in_path), cv2.IMREAD_GRAYSCALE)
    out = operations[args.operation](img)
    cv2.imwrite(str(out_path), out)


if __name__ == "__main__":
    main()
