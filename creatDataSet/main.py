import cv2 as cv
import time

dataPath = "img/"


def MatrixToImage(data, filename):
    cv.imwrite(dataPath + filename + ".jpg", data, [int(cv.IMWRITE_JPEG_QUALITY), 100])


if __name__ == "__main__":
    cap = cv.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            frame = cv.flip(frame, 1)
            frame = frame[:, :320, :]  # 右摄像头
            cv.imshow('Video Record', frame)
            key = cv.waitKey(1)
            if key == 32:
                t = time.time()
                MatrixToImage(frame, str(int(t)))
                print("shoot done!")

            elif key == 27:
                break
cv.destroyAllWindows()
