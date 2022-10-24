import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image


def normalize(value):
    if value < 0:
        return 0
    elif value > 255:
        return 255
    else:
        return value


def find_green(cell, i=0):
    N = abs(cell[2, 2, i] - cell[0, 2, i]) * 2 + abs(cell[1, 2, 1] - cell[3, 2, 1])
    E = abs(cell[2, 2, i] - cell[2, 4, i]) * 2 + abs(cell[2, 1, 1] - cell[2, 3, 1])
    W = abs(cell[2, 2, i] - cell[2, 0, i]) * 2 + abs(cell[2, 1, 1] - cell[2, 3, 1])
    S = abs(cell[2, 2, i] - cell[4, 2, i]) * 2 + abs(cell[1, 2, 1] - cell[3, 2, 1])

    argmin = np.argmin([N, E, W, S])
    if argmin == 0:
        return (cell[1, 2, 1] * 3 + cell[3, 2, 1] + cell[2, 2, i] - cell[0, 2, i]) / 4
    elif argmin == 1:
        return (cell[2, 3, 1] * 3 + cell[2, 1, 1] + cell[2, 2, i] - cell[2, 4, i]) / 4
    elif argmin == 2:
        return (cell[2, 1, 1] * 3 + cell[2, 3, 1] + cell[2, 2, i] - cell[2, 0, i]) / 4
    else:
        return (cell[3, 2, 1] * 3 + cell[1, 2, 1] + cell[2, 2, i] - cell[4, 2, i]) / 4


def first_stage(cell, bayer_filter, itog, i, j):
    if bayer_filter[0, 0] == 'r':
        value = find_green(cell)
        itog[i - 2, j - 2, 1] = normalize(value)
    elif bayer_filter[0, 0] == 'b':
        value = find_green(cell, 2)
        itog[i - 2, j - 2, 1] = normalize(value)


def hue_transit(L1, L2, L3, V1, V3):
    if L1 < L2 < L3 or L1 > L2 > L3:
        return V1 + (V3 - V1) * (L2 - L1) / (L3 - L1)
    else:
        return (V1 + V3) / 2 + (L2 - (L1 + L3) / 2) / 2


def find_red_and_blue(cell, swap):
    i, j = 0, 2
    if swap:
        i, j = j, i
    r = hue_transit(cell[1, 2, 1], cell[2, 2, 1], cell[3, 2, 1], cell[1, 2, i], cell[3, 2, i])
    b = hue_transit(cell[2, 1, 1], cell[2, 2, 1], cell[2, 3, 1], cell[2, 1, j], cell[2, 3, j])
    if swap:
        r, b = b, r
    return r, b


def second_stage(cell, bayer_filter, itog, i, j):
    if bayer_filter[0, 0] == 'g':
        if bayer_filter[1, 0] == 'r':
            r, b = find_red_and_blue(cell, False)
            itog[i - 2, j - 2, (0, 2)] = normalize(r), normalize(b)
        else:
            r, b = find_red_and_blue(cell, True)
            itog[i - 2, j - 2, (0, 2)] = normalize(r), normalize(b)


def find_red_or_blue(cell, swap):
    i, j = 0, 2
    if swap:
        i, j = j, i

    NE = abs(cell[1, 3, j] - cell[3, 1, j]) + abs(cell[0, 4, i] - cell[2, 2, i]) \
         + abs(cell[2, 2, i] - cell[4, 0, i]) + abs(cell[1, 3, 1] - cell[2, 2, 1]) \
         + abs(cell[3, 1, 1] - cell[2, 2, 1])
    NW = abs(cell[1, 3, j] - cell[3, 1, j]) + abs(cell[0, 0, i] - cell[2, 2, i]) \
         + abs(cell[2, 2, i] - cell[4, 4, i]) + abs(cell[1, 1, 1] - cell[2, 2, 1]) \
         + abs(cell[3, 3, 1] - cell[2, 2, 1])

    if NE < NW:
        return hue_transit(cell[1, 3, 1], cell[2, 2, 1], cell[3, 1, 1], cell[1, 3, j], cell[3, 1, j])
    else:
        return hue_transit(cell[1, 1, 1], cell[2, 2, 1], cell[3, 3, 1], cell[1, 1, j], cell[3, 3, j])


def third_stage(cell, bayer_filter, itog, i, j):
    if bayer_filter[0, 0] == 'r':
        value = find_red_or_blue(cell, False)
        itog[i - 2, j - 2, 2] = normalize(value)
    elif bayer_filter[0, 0] == 'b':
        value = find_red_or_blue(cell, True)
        itog[i - 2, j - 2, 0] = normalize(value)


def interpolation(padded_image, itog, bayer_filter, stage):
    for i in range(2, padded_image.shape[0] - 2):
        for j in range(2, padded_image.shape[1] - 2):
            cell = padded_image[i - 2:i + 3, j - 2:j + 3]
            stage(cell, bayer_filter, itog, i, j)
            bayer_filter[:, (0, 1)] = bayer_filter[:, (1, 0)]

        bayer_filter[(0, 1), :] = bayer_filter[(1, 0), :]


def reconstruct(image, bayer_filter):
    padded_image = np.concatenate((image[:2, :][::-1, :, :], image))
    padded_image = np.concatenate((padded_image, padded_image[-3:, :][::-1, :, :]))
    padded_image = np.concatenate((padded_image[:, :2][:, ::-1, :], padded_image), axis=1)
    padded_image = np.concatenate((padded_image, padded_image[:, -2:][:, ::-1, :]), axis=1)
    padded_image = padded_image.astype(float)

    itog = image.copy().astype(float)
    itog = np.concatenate((itog, itog[-1:, :]))

    interpolation(padded_image, itog, bayer_filter, first_stage)
    interpolation(padded_image, itog, bayer_filter, second_stage)
    interpolation(padded_image, itog, bayer_filter, third_stage)

    itog = itog[:-1, :]
    itog[itog < 0] = 0
    itog[itog > 255] = 255
    itog = itog.astype(np.uint8)

    return itog


def int_array(image):
    a = np.ndarray((image.shape[0], image.shape[1], image.shape[2]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(image.shape[2]):
                a[i][j][k] = int(image[i][j][k])

    return a


def main():
    image = plt.imread('RGB_CFA.bmp')

    bayer_filter = np.array([['r', 'g'], ['g', 'b']])
    start_time = time.time()
    itog = reconstruct(image, bayer_filter)
    time_work = time.time() - start_time

    im = Image.fromarray(itog.astype(np.uint8))
    im.save('itog.bmp')

    original = plt.imread('Original.bmp')
    megapixel_count = original.shape[0] * original.shape[1] / 100000

    int_original = int_array(original)
    int_itog = int_array(itog)

    PSNR = 10 * np.log10(255 ** 2 / (np.sum((int_original - int_itog) ** 2) / (original.shape[0] * original.shape[1])))
    print("PSNR:", PSNR)   # 14.80275814586986
    print("Time:", time_work, "sec")  # 221.69114804267883
    print(time_work / megapixel_count, "sec/megapixel")  # 2.560876060694133


if __name__ == "__main__":
    main()
