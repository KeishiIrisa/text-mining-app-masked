import cv2
import numpy as np


def masking(fg_img):
    # hsv変換
    hsv = cv2.cvtColor(fg_img, cv2.COLOR_BGR2HSV)

    # 2値化
    bin_img = cv2.inRange(hsv, (0, 10, 0), (255, 255, 255))

    # 輪郭抽出
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積が最大の輪郭を取得する
    contour = max(contours, key=lambda x: cv2.contourArea(x))

    # マスク画像を作成する。
    mask = np.zeros_like(bin_img)
    cv2.drawContours(mask, [contour], -1, color=255, thickness=-1)

    # マスク画像の白黒を反転させる
    mask = cv2.bitwise_not(mask)
   
    return mask
