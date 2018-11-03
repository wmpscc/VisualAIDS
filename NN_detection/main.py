import json
import time
import argparse
import cv2
import numpy as np
import mvnc.mvncapi as mvncapi
import movidus_utils
import yolo_utils


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

    cap.release()


if __name__ == "__main__":
    inference_video()
