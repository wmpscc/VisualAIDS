import json
import time
import argparse
import cv2
import numpy as np
import mvnc.mvncapi as mvncapi
import movidus_utils
import yolo_utils
import math
import ace


def stretchImage(data, s=0.005, bins=2000):  # 线性拉伸，去掉最大最小0.5%的像素值，然后线性拉伸至[0,1]
    ht = np.histogram(data, bins)
    d = np.cumsum(ht[0]) / float(data.size)
    lmin = 0
    lmax = bins - 1
    while lmin < bins:
        if d[lmin] >= s:
            break
        lmin += 1
    while lmax >= 0:
        if d[lmax] <= 1 - s:
            break
        lmax -= 1
    return np.clip((data - ht[1][lmin]) / (ht[1][lmax] - ht[1][lmin]), 0, 1)


g_para = {}


def getPara(radius=5):  # 根据半径计算权重参数矩阵
    global g_para
    m = g_para.get(radius, None)
    if m is not None:
        return m
    size = radius * 2 + 1
    m = np.zeros((size, size))
    for h in range(-radius, radius + 1):
        for w in range(-radius, radius + 1):
            if h == 0 and w == 0:
                continue
            m[radius + h, radius + w] = 1.0 / math.sqrt(h ** 2 + w ** 2)
    m /= m.sum()
    g_para[radius] = m
    return m


def zmIce(I, ratio=4, radius=300):  # 常规的ACE实现
    para = getPara(radius)
    height, width = I.shape
    zh, zw = [0] * radius + list(range(height)) + [height - 1] * radius, [0] * radius + list(range(width)) + [
        width - 1] * radius
    Z = I[np.ix_(zh, zw)]
    res = np.zeros(I.shape)
    for h in range(radius * 2 + 1):
        for w in range(radius * 2 + 1):
            if para[h][w] == 0:
                continue
            res += (para[h][w] * np.clip((I - Z[h:h + height, w:w + width]) * ratio, -1, 1))
    return res


def zmIceFast(I, ratio, radius):  # 单通道ACE快速增强实现
    height, width = I.shape[:2]

    if min(height, width) <= 2:
        return np.zeros(I.shape) + 0.5
    Rs = cv2.resize(I, (np.int((width + 1) / 2), np.int((height + 1) / 2)))
    Rf = zmIceFast(Rs, ratio, radius)  # 递归调用
    Rf = cv2.resize(Rf, (width, height))
    Rs = cv2.resize(Rs, (width, height))

    return Rf + zmIce(I, ratio, radius) - zmIce(Rs, ratio, radius)


def zmIceColor(I, ratio=2, radius=3):  # rgb三通道分别增强，ratio是对比度增强因子，radius是卷积模板半径
    res = np.zeros(I.shape)
    for k in range(3):
        res[:, :, k] = stretchImage(zmIceFast(I[:, :, k], ratio, radius))
    return res



def inference_image(dev,
                    graph_file,
                    meta_file,
                    img_in,
                    threshold=0.3):
    meta = yolo_utils.get_meta(meta_file)
    meta['thresh'] = threshold

    graph, input_fifo, output_fifo = movidus_utils.load_graph(dev, graph_file)
    img = img_in

    img = img.astype(np.float32)
    img_orig = np.copy(img)
    img_orig_dimensions = img_orig.shape
    img = yolo_utils.pre_proc_img(img, meta)
    graph.queue_inference_with_fifo_elem(
        input_fifo, output_fifo, img, 'user object')
    output, _ = output_fifo.read_elem()
    y_out = np.reshape(output, (13, 13, 30))
    y_out = np.squeeze(y_out)
    boxes = yolo_utils.procces_out(y_out, meta, img_orig_dimensions)
    yolo_utils.add_bb_to_img(img_orig, boxes)
    return img_orig, boxes


def get_cap():
    cap = cv2.VideoCapture(1)
    dev = movidus_utils.get_mvnc_device()

    while cap.isOpened():
        ret, frame = cap.read()
        cap.set(3, 640)
        cap.set(4, 240)
        cap.set(cv2.CAP_PROP_FPS, 4)

        if ret == True:
            frame = frame[:, :320, :]
            img, boxes = inference_image(dev=dev,
                                         graph_file="graph/tiny-yolo-voc-1c.graph",
                                         meta_file="graph/tiny-yolo-voc-1c.meta",
                                         img_in=frame, threshold=0.3)
            cv2.imshow("img", img)
            print(boxes)

            k = cv2.waitKey(1) & 0xff




def inference_video(graph_file="graph/tiny-yolo-voc-1c.graph",
                    meta_file="graph/tiny-yolo-voc-1c.meta",
                    threshold=0.3):
    meta = yolo_utils.get_meta(meta_file)
    meta['thresh'] = threshold
    dev = movidus_utils.get_mvnc_device()
    graph, input_fifo, output_fifo = movidus_utils.load_graph(dev, graph_file)
    cap = cv2.VideoCapture(1)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(video_out_name, fourcc, fps, (width, height))
    times = []
    while True:
        ret, frame = cap.read()
        frame = frame[:, :320, :]
        frame = zmIceColor(frame / 255.0) * 255     # ACE色彩增强
        frame = frame.astype(np.uint8)

        if not ret:
            print("Video Ended")
            break
        frame_orig = np.copy(frame)
        img_orig_dimensions = frame_orig.shape
        frame = yolo_utils.pre_proc_img(frame, meta)
        start = time.time()
        graph.queue_inference_with_fifo_elem(
            input_fifo, output_fifo, frame, 'user object')
        output, _ = output_fifo.read_elem()
        end = time.time()
        print('FPS: {:.2f}'.format((1 / (end - start))))
        times.append((1 / (end - start)))
        y_out = np.reshape(output, (13, 13, 30))
        y_out = np.squeeze(y_out)
        boxes = yolo_utils.procces_out(y_out, meta, img_orig_dimensions)
        yolo_utils.add_bb_to_img(frame_orig, boxes)
        cv2.imshow("out", frame_orig)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cap.release()


if __name__ == "__main__":
    inference_video()
