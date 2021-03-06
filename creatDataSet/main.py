import cv2 as cv
import time
import numpy as np

dataPath = "cur/"


def MatrixToImage(data, filename):
    cv.imwrite(dataPath + filename + ".jpg", data, [int(cv.IMWRITE_JPEG_QUALITY), 100])


if __name__ == "__main__":
    cap = cv.VideoCapture(1)
    # cap.set(3, 640)
    # cap.set(4, 240)
    # cap.set(3, 1280)
    # cap.set(4, 480)


    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            # frame = cv.flip(frame, 1)
            # frame = frame[:416, :416, :416]  # 右摄像头

            l_frame = frame[:, :320, :]
            r_frame = frame[:, 320:, :]
            # print(np.shape(l_frame))


            cv.imshow('Video Record  L', l_frame)
            cv.imshow('Video Record  R', r_frame)

            key = cv.waitKey(1)
            if key == 32:
                t = time.time()
                # MatrixToImage(frame, str(int(t)))
                MatrixToImage(l_frame, 'L' + str(int(t)))
                MatrixToImage(r_frame, 'R' + str(int(t)))

                print("shoot done!")

            elif key == 27:
                break
cv.destroyAllWindows()
