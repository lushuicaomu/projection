import cv2
import numpy as np
import matplotlib.pyplot as plt


def FOV2f(fov, diagnoal):

    f = diagnoal / (2 * np.tan(fov / 2))
    return f

def correct(image, fov):

    h, w, _ = image.shape
    d = max(h, w)
    f = FOV2f(fov, d)

    x = (np.arange(0, w, 1) - w / 2).astype(np.float32)
    y = (np.arange(0, h, 1) - h / 2).astype(np.float32)
    x, y = np.meshgrid(x, y)


    # forward
    # l = np.sqrt(np.power(x, 2) + np.power(f, 2))
    # phi = np.arctan(x / f)
    # theta = np.arctan(y / l)
    # x = phi * f + w / 2
    # y = f * np.tan(theta) + h / 2

    # backward
    l = np.sqrt(np.power(x, 2) + np.power(f, 2))
    x = f * np.tan(x / f) + w / 2
    y = y * l / f + h / 2

    out = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)

    return out

if __name__ == "__main__":

    image = cv2.imread("./../data/1_97.jpg")[::1, ::1, :]
    out = correct(image, np.pi * 97 / 180)
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, ::-1])
    plt.title("image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(out[:, :, ::-1])
    plt.title("out")
    plt.axis("off")
    plt.show()
