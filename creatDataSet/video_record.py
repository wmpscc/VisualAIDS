import cv2
import numpy as np
import time

dataPath = "video/"

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 480)
    t = str(int(time.time()))
    out = cv2.VideoWriter(dataPath + t + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (1280, 480))
    while True:
        # 一帧一帧的获取图像
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.flip(frame, 1)
            # 在帧上进行操作
            # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            # 开始保存视频
            out.write(frame)
            # 显示结果帧
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    # 释放摄像头资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
