import numpy as np
import cv2
import camera_configs


class DisparityEstimation():
    def eismation(self, camera1):
        # camera1 = cv2.VideoCapture(0)
        # camera1.set(3, 640)
        # camera1.set(4, 480)
        # while True:
        for k in range(300):

            try:
                _, frame = camera1.read()
                if k % 3 != 0:
                    continue
                frame1 = frame[:240, :320, :]  # 左
                frame2 = frame[:240, 320:640, :]  # 右
                # 根据更正map对图片进行重构
                img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR)
                img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR)
                # 将图片置为灰度图，为StereoBM作准备
                imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
                imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
                # 两个trackbar用来调节不同的参数查看效果
                num = 0
                blockSize = 30
                if blockSize % 2 == 0:
                    blockSize += 1
                if blockSize < 5:
                    blockSize = 5
                # 根据Block Maching方法生成差异图（opencv里也提供了SGBM/Semi-Global Block Matching算法，有兴趣可以试试）
                stereo = cv2.StereoBM_create(numDisparities=16 * num, blockSize=blockSize)
                disparity = stereo.compute(imgL, imgR)

                disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                # 将图片扩展至3d空间中，其z方向的值则为当前的距离
                threeD = cv2.reprojectImageTo3D(disparity.astype(np.float32) / 16., camera_configs.Q)
                # halfDis = disp.shape[1] / 2
                halfDis = 210
                # print("half:",halfDis)
                for i in range(0, disp.shape[0], 25):
                    for j in range(0, (disp.shape[1]), 25):
                        if threeD[i][j][2] > 1500.0 or threeD[i][j][2] < 0.0:
                            disp[i][j] = 0
                        if threeD[i][j][2] >= 500.0 and threeD[i][j][2] <= 1500.0:
                            disp[i][j] = 255
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                disp = cv2.dilate(disp, kernel)
                _, disp = cv2.threshold(disp, 200, 255, cv2.THRESH_BINARY)
                _, contours, _ = cv2.findContours(disp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnt = contours[1]
                area = cv2.contourArea(cnt)
                if area > 50.0:
                    cntM = contours[0]
                    M = cv2.moments(cntM)
                    cx = int(M['m10'] / M['m00'])
                    cv2.drawContours(disp, contours, -1, (100, 255, 125), 3)
                    reSt = "前方有障碍物"
                    if (cx - halfDis) / cx > 0.5:
                        reDis = "障碍物位于右边"
                        # print("test",cx, halfDis,cx-halfDis, (cx - halfDis) / cx)
                        return reSt, reDis
                    if (cx - halfDis) / cx < 0.5:
                        reDis = "障碍物位于左边"
                        # print("test", cx, halfDis, cx - halfDis, (cx - halfDis) / cx)
                        return reSt, reDis
                    # k = cv2.waitKey(1) & 0xff
                # else:
                # print(k)


            except:
                print("fin cash")
        reSt = "前方无障碍物"
        reDis = "可以放心前行"
        return reSt, reDis

# reSt = "前方没有障碍物"
# reDis = ""
# return reSt, reDis
        # camera1.release()
