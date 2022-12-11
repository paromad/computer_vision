import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import sobel


def roll_columns(matrix, r):
    row_indices, columns = np.ogrid[:matrix.shape[0], :matrix.shape[1]]
    r[r < 0] += matrix.shape[1]
    row_indices = row_indices - r
    result = matrix[row_indices, columns]
    return result

def calc_sums(image, xmin, xmax):
    result = np.zeros((image.shape[0], xmax - xmin + 1))

    if xmax - xmin == 1:
        result[:, 0] = image[:, xmin]
    else:
        mid = (xmax + xmin) // 2
        left_column = calc_sums(image, xmin, mid)
        right_column = calc_sums(image, mid, xmax)

        shift = np.arange(xmax - xmin + 1).astype(int)
        rolled_matrix = roll_columns(right_column[:, shift // 2 - shift % 2], shift // 2 + shift % 2)

        result[:, shift] = left_column[:, shift // 2] + rolled_matrix

    return result

def fht(image):
    hough_space = calc_sums(image, 0, image.shape[1])
    hough_space_hor = calc_sums(image[:, ::-1], 0, image.shape[1])

    return np.hstack([hough_space[:, ::-1], hough_space_hor])


def get_angles(path):
    names = os.listdir(path)
    angles = np.zeros(len(names))
    for i, name in enumerate(names):
        image = plt.imread(os.path.join(path, name))

        hough_space = fht(sobel(rgb2gray(image)))
        angles[i] = np.arctan((hough_space.var(axis=0).argmax() - hough_space.shape[1] // 2) / image.shape[1]) * 180.0 / np.pi

    return angles

def rotate(image, angle, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE):
    W, H = image.shape[:2]

    center = (W // 2, H // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (H, W), flags=flags, borderMode=borderMode)

    return rotated

def main(in_dir_path = './images', out_dir_path = './itog'):
    names = os.listdir(in_dir_path)
    angles = get_angles(in_dir_path)

    for angle, name in zip(angles, names):
        image = plt.imread(os.path.join(in_dir_path, name))

        rotated = rotate(image, angle, flags=cv2.INTER_NEAREST)
        im = Image.fromarray(rotated.astype(np.uint8))
        im.save('./itog/nearest/{}'.format(name))

        rotated = rotate(image, angle, flags=cv2.INTER_LINEAR)
        im = Image.fromarray(rotated.astype(np.uint8))
        im.save('./itog/linear/{}'.format(name))


if __name__ == "__main__":
    main()
