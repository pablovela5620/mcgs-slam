import os
import re
import numpy as np
import cv2
import torch

base = 8


def map_filename(x):
    return float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1])


def resize(image, h1, w1):
    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1 % base, :w1-w1 % base]
    image = torch.as_tensor(image).permute(2, 0, 1)
    return image


def image_stream(imagedirs, calib, args):
    """ image generator """
    image_lists = []
    for imagedir in imagedirs:
        image_lists.append(sorted(os.listdir(imagedir), key=map_filename)[::args.stride])

    h1, w1 = args.ht, args.wd
    for t in range(len((image_lists[0]))):
        images = []
        for i, imagelist in enumerate(image_lists):
            timestamp = float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", imagelist[t])[-1]) / args.timescale
            if i == 0:
                t0 = timestamp
            else:
                assert abs(timestamp - t0) < 2e-2, f"{timestamp} vs. {t0}"

            image = cv2.imread(os.path.join(imagedirs[i], imagelist[t]))
            
            # TODO
            if calib.shape[1] > 4:
                K = np.array([[calib[i][0], 0, calib[i][2]], [0, calib[i][1], calib[i][3]], [0, 0, 1]])
                image = cv2.undistort(image, K, calib[i][4:])
            
            h0, w0, _ = image.shape
            image = resize(image, h1, w1)
            images.append(image)

        images = torch.stack(images, dim=0)

        intrinsics = torch.zeros((len(imagedirs), 8))
        intrinsics[:, :4] = torch.tensor(calib[:, :4])

        intrinsics[:, [0, 2]] *= (w1 / w0)
        intrinsics[:, [1, 3]] *= (h1 / h0)

        if t == 0:
            print(
                f'Orig size: {h0}-{w0}; Down to size: {h1}-{w1}\nInput calib: {list(calib)}\nAdapt calib: {list(intrinsics.numpy())}')

        yield t, images, intrinsics, timestamp
