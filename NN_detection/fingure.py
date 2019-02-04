import numpy as np
import cv2 as cv

## Defining Constants
lwr = np.array([0, 50, 70], np.uint8)
upr = np.array([100, 230, 230], np.uint8)
kernel = np.ones((5, 5), np.uint8)
fingerP = []
fingerT = []
st = False
p = False


## HSV function to separate hand(skin color) from background(color)
def hsvF(focusF):
    hsv = cv.cvtColor(focusF, cv.COLOR_BGR2HSV_FULL)
    mask = cv.inRange(hsv, lwr, upr)
    mask = cv.dilate(mask, kernel, iterations=3)
    mask = cv.GaussianBlur(mask, (5, 5), 100)
    return mask


## Function to mark the centroid of contour
def centroidF(cnt):
    M = cv.moments(cnt)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return cx, cy
    else:
        pass


## Function to display the topmost point of the contour
def points(focusF, finger):
    for i in range(len(finger)):
        cv.circle(focusF, finger[i], 5, [182, 31, 102], -1)


## Function to pause/start the tracking of the finger
# Press 'p' to pause/start the tracking
# Press 'c' to clear the screen
def pauseT(focusF, fingerT, ftop):
    if len(fingerT) < 20:
        fingerT.append(ftop)
    else:
        fingerT.pop(0)
        fingerT.append(ftop)
    points(focusF, fingerT)


def get_fingure_top(focusF, are_threshold=3000):
    noise = hsvF(focusF)

    r, thresh = cv.threshold(noise, 100, 255, cv.THRESH_BINARY)
    img, cont, hie = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = max(cont, key=cv.contourArea)

    epsilon = 0.001 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)

    hull = cv.convexHull(approx, returnPoints=False)
    area_cnt = cv.contourArea(approx)
    defects = cv.convexityDefects(approx, hull)

    centroid = centroidF(cnt)

    if defects is not None:
        if area_cnt > are_threshold:  # 轮廓面积
            print("--------")
            ftop = tuple(cnt[cnt[:, :, 1].argmin()][0])
            cv.circle(focusF, ftop, 5, (0, 0, 255), -1)  # 指尖
            print(ftop)
            cv.circle(focusF, centroid, 3, (0, 255, 255), -1)  # 中心点
    return ftop

