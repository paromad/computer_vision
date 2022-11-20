import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image


def median(image, x, y, R = 1):
    res = []
    for c in range(image.shape[2]):
        arr = []
        for i in range(2 * R + 1):
            cell_x = x - R + i
            for j in range(2 * R + 1):
                cell_y = y - R + j
                if cell_x < 0 or cell_x >= image.shape[0] or cell_y < 0 or cell_y >= image.shape[1]:
                    arr.append(0)
                else:
                    arr.append(image[cell_x][cell_y][c])

        arr = sorted(arr)
        res.append(arr[len(arr) // 2])

    return res


def simple_sort(image, R = 1):
    res_image = np.zeros(image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            res_image[x][y] = median(image, x, y, R)

    return res_image


def huang(image, R = 1):
    res_image = np.zeros(image.shape)

    pad_image = np.concatenate((image[:R, :], image))
    pad_image = np.concatenate((pad_image, pad_image[-R:, :]))
    pad_image = np.concatenate((pad_image[:, :R], pad_image), axis=1)
    pad_image = np.concatenate((pad_image, pad_image[:, -R:]), axis=1)

    for c in range(image.shape[2]):
        for x in range(image.shape[0]):
            w = pad_image[x: x + 2 * R + 1, :R * 2 + 1, c].reshape(-1)
            counts = np.zeros(256)
            counts[:len(np.bincount(w))] += np.bincount(w)
            res_image[x, 0, c] = np.where(np.cumsum(counts) > len(w) // 2)[0][0]

            for y in range(1, image.shape[1]):
                p = pad_image[x: x + 2 * R + 1, y - 1, c]
                counts[:len(np.bincount(p))] -= np.bincount(p)
                p = pad_image[x: x + 2 * R + 1, y + 2 * R, c]
                counts[:len(np.bincount(p))] += np.bincount(p)
                res_image[x, y, c] = np.where(np.cumsum(counts) > len(w) // 2)[0][0]

    return res_image


def const_time(image, R = 2):
    res_image = np.zeros(image.shape)

    pad_image = np.concatenate((image[:R, :], image))
    pad_image = np.concatenate((pad_image, pad_image[-R:, :]))
    pad_image = np.concatenate((pad_image[:, :R], pad_image), axis=1)
    pad_image = np.concatenate((pad_image, pad_image[:, -R:]), axis=1)

    for c in range(image.shape[2]):
        column_histograms = np.array([np.bincount(pad_image[:2 * R, i, c], minlength=256) for i in range(pad_image.shape[1])])

        for x in range(res_image.shape[0]):
            w = pad_image[x: x + 2 * R + 1, : 2 * R + 1, c].reshape(-1)
            counts = np.zeros(256, dtype=int)
            counts[:len(np.bincount(w))] += np.bincount(w)
            res_image[x, 0, c] = np.where(np.cumsum(counts) > len(w) // 2)[0][0]

            for y in range(res_image.shape[1] + 2 * R):
                if x > 0:
                    column_histograms[y, pad_image[x - 1, y, c]] -= 1
                column_histograms[y, pad_image[x + 2 * R, y, c]] += 1

                if y >= 2 * R + 1:
                    counts -= column_histograms[y - 2 * R - 1]
                    counts += column_histograms[y]
                    res_image[x, y - 2 * R, c] = np.where(np.cumsum(counts) > len(w) // 2)[0][0]

    return res_image


def main(in_name = 'picture.bmp', out_name = 'itog_2.bmp', R = 2):
    image = plt.imread(in_name)
    start_time = time.time()

    itog = const_time(image, R)
    time_work = time.time() - start_time
    print(time_work)

    im = Image.fromarray(itog.astype(np.uint8))
    im.save(out_name)


if __name__ == "__main__":
    main()
