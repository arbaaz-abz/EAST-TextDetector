import cv2
import numpy as np


def main(box_coordinates, image):
    pts = np.array(box_coordinates)
    rect = cv2.boundingRect(pts)

    x, y, w, h = rect
    croped = image[y:y + h, x:x + w].copy()

    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(croped, croped, mask=mask)

    bg = np.ones_like(croped, np.uint8) * 255
    cv2.bitwise_not(bg, bg, mask=mask)
    dst2 = bg + dst

    return dst2
