import cv2
import numpy as np

left_camera_matrix = np.array([[186.41610, 0., 154.47008],
                               [0., 185.82732, 129.22643],
                               [0., 0., 1.]])
left_distortion = np.array([[-0.07087 , 0.05282 , 0.00317 , 0.00394 , 0.00000]])



right_camera_matrix = np.array([[180.67416, 0., 156.75899],
                                [0., 180.31310, 126.39907],
                                [0., 0., 1.]])
right_distortion = np.array([[ -0.03570 , 0.02850 , 0.00527 , -0.00452 , 0.00000]])

om = np.array([-0.01465 , -0.03053 , 0.00310]) # 旋转关系向量
R = cv2.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
T = np.array([-142.81669 , 3.31789 , -12.05164]) # 平移关系向量




size = (640, 240) # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,right_camera_matrix, right_distortion, size, R,T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)