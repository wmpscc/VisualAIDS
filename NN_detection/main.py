import json
import time
import cv2
import numpy as np
import movidus_utils
import yolo_utils
from graph_ace import ace_algorithm
from fingure_detection import fingure_top
from Object_Detection import object
ace = ace_algorithm()
fd = fingure_top()
obd = object()


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





def inference_video(graph_file="graph/tiny-yolo-voc-1c.graph",
                    meta_file="graph/tiny-yolo-voc-1c.meta",
                    threshold=0.3):
    meta = yolo_utils.get_meta(meta_file)
    meta['thresh'] = threshold
    dev = movidus_utils.get_mvnc_device()
    graph, input_fifo, output_fifo = movidus_utils.load_graph(dev, graph_file)
    is_open = True

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
        # frame = ace.ace(frame)    # ACE色彩增强
        frame = frame.astype(np.uint8)

        if not ret:
            print("Video Ended")
            break
        frame_orig = np.copy(frame)
        img_orig_dimensions = frame_orig.shape
        frame = yolo_utils.pre_proc_img(frame, meta)
        start = time.time()

        if is_open == False:
            dev = movidus_utils.get_mvnc_device()
            graph, input_fifo, output_fifo = movidus_utils.load_graph(dev, graph_file)

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

        if(len(boxes) != 0):
            box = boxes[0]
            print(box)
            left = box["topleft"]['x']
            right = box["bottomright"]['x']
            top = box["topleft"]['y']
            bot = box["bottomright"]['y']
            print(top, bot, left, right)
            roi = frame_orig[top:bot, left:right, :3]
            ftop = fd.get_fingure_top(roi)
            if ftop != -1:
                cv2.circle(roi, ftop, 5, (0, 0, 255), -1)
                cv2.imshow("roi", roi)

                ftop = list(ftop)


                ftop[0] = ftop[0] + left
                ftop[1] = ftop[1] + top
                ftop = tuple(ftop)
                cv2.circle(frame_orig, ftop, 5, (0, 0, 255), -1)
                print(ftop)
                input_fifo.destroy()
                output_fifo.destroy()
                graph.destroy()
                dev.close()
                dev.destroy()
                is_open = False
                print("out:" + obd.calc_target(ftop, frame_orig))
        cv2.imshow("out", frame_orig)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cap.release()


if __name__ == "__main__":
    inference_video()
