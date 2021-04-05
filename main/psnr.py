import cv2
import utils
import os
import glob
import math
import numpy as np

pred_Path = ""


def calc_psnr(img1, img2):
    img1 = img1 / 255.0
    img2 = img2 / 255.0
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def calc_averger_psnr(pred_Path):
    file_list = os.listdir(pred_Path)
    PSNR = 0
    length = len(file_list)
    for file_name in file_list:
        test_list = glob.glob(pred_Path + file_name)
        for pred in test_list:
            # file_name_name = file_name[2:] # work at test status
            gt = "./data/SICE_val/normal/" + file_name

            pred_img = cv2.imread(pred)
            gt_img = cv2.imread(gt)

            psnr = calc_psnr(pred_img, gt_img)
            PSNR = PSNR + psnr

    print(PSNR / length)
    return PSNR / length


if __name__ == '__main__':
    result = calc_averger_psnr(pred_Path)
