import cv2
import numpy as np
import matplotlib.pyplot as plt


def FOV2f(fov, diagnoal):

    f = diagnoal / (2 * np.tan(fov / 2))
    return f

def correctV1(image, fov):

    h, w, _ = image.shape
    d = min(h, w)
    f = FOV2f(fov, d)

    x = (np.arange(0, w, 1) - w / 2).astype(np.float32)
    y = (np.arange(0, h, 1) - h / 2).astype(np.float32)
    x, y = np.meshgrid(x, y)

    coords = np.stack([x, y], axis=-1)
    rp = np.linalg.norm(coords, axis=-1)
    r0 = d / (2 * np.tan(0.5 * np.arctan(d / (2 * f))))
    ru = r0 * np.tan(0.5 * np.arctan(rp / f))

    x = x / ru * rp + w / 2
    y = y / ru * rp + h / 2

    out = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)

    return out

def correctV2(image, fov):

    h, w, _ = image.shape
    d = max(h, w)
    f = FOV2f(fov, d)

    x = (np.arange(0, w, 1) - w / 2).astype(np.float32)
    y = (np.arange(0, h, 1) - h / 2).astype(np.float32)
    x, y = np.meshgrid(x, y)

    coords = np.stack([x, y], axis=-1)
    rp = np.linalg.norm(coords, axis=-1)
    ru = 2 * f * np.tan(0.5 * np.arctan(rp / f))

    # forward
    # x = x / rp * ru + w / 2
    # y = y / rp * ru + h / 2

    # backward
    x = x / ru * rp + w / 2
    y = y / ru * rp + h / 2

    out = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)

    return out

if __name__ == "__main__":

    image = cv2.imread("./../data/1_97.jpg")[::1, ::1, :]
    outV1 = correctV1(image, np.pi * 97 / 180)
    outV2 = correctV2(image, np.pi * 97 / 180)
    plt.subplot(1, 3, 1)
    plt.imshow(image[:, :, ::-1])
    plt.title("image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(outV1[:, :, ::-1])
    plt.title("outV1")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(outV2[:, :, ::-1])
    plt.title("outV2")
    plt.axis("off")
    plt.show()
