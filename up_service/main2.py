import socket
import json
import cv2
import numpy as np
import movidus_utils
import yolo_utils
from graph_ace import ace_algorithm
from fingure_detection import fingure_top
from Object_Detection import object
from FinalCode import DisparityEstimation

ace = ace_algorithm()
fd = fingure_top()
obd = object()
estimation = DisparityEstimation()


def get_fingure_label(graph_file="graph/tiny-yolo-voc-1c.graph",
                      meta_file="graph/tiny-yolo-voc-1c.meta",
                      threshold=0.2):
    meta = yolo_utils.get_meta(meta_file)
    meta['thresh'] = threshold
    dev = movidus_utils.get_mvnc_device()
    graph, input_fifo, output_fifo = movidus_utils.load_graph(dev, graph_file)
    is_open = True

    cap = cv2.VideoCapture(0)

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

        if is_open == False:
            dev = movidus_utils.get_mvnc_device()
            graph, input_fifo, output_fifo = movidus_utils.load_graph(dev, graph_file)

        graph.queue_inference_with_fifo_elem(
            input_fifo, output_fifo, frame, 'user object')
        output, _ = output_fifo.read_elem()
        y_out = np.reshape(output, (13, 13, 30))
        y_out = np.squeeze(y_out)
        boxes = yolo_utils.procces_out(y_out, meta, img_orig_dimensions)
        yolo_utils.add_bb_to_img(frame_orig, boxes)

        if (len(boxes) != 0):
            box = boxes[0]
            # print(box)
            left = box["topleft"]['x']
            right = box["bottomright"]['x']
            top = box["topleft"]['y']
            bot = box["bottomright"]['y']
            # print(top, bot, left, right)
            roi = frame_orig[top:bot, left:right, :3]
            ftop = fd.get_fingure_top(roi)
            if ftop != -1:
                cv2.circle(roi, ftop, 5, (0, 0, 255), -1)
                ftop = list(ftop)
                ftop[0] = ftop[0] + left
                ftop[1] = ftop[1] + top
                ftop = tuple(ftop)
                input_fifo.destroy()
                output_fifo.destroy()
                graph.destroy()
                # dev.close()
                # dev.destroy()
                return obd.calc_target(ftop, frame_orig, dev)
        # cv2.imshow("out", frame_orig)
        k = cv2.waitKey(1) & 0xff

    cap.release()


# def getip():
#     try:
#         s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         s.connect(("www.baidu.com", 0))
#         ip = s.getsockname()[0]
#         print(ip)
#     except:
#         ip = "x.x.x.x"
#     finally:
#         s.close()
#     return ip
#
#
# print("start socket: TCP...")
# socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#
# HOST_IP = getip()
# HOST_PORT = 7653
#
# print("tcp server listen @ %s:%d!" % (HOST_IP, HOST_PORT))
# host_addr = (HOST_IP, HOST_PORT)
# socket_tcp.bind(host_addr)
# socket_tcp.listen(1)

# while True:
#     print("waiting for connection...")
#     socket_con, (client_ip, client_port) = socket_tcp.accept()
#     print("connectionaccepted from %s." % client_ip)
#
#     # socket_con.send(b"welcome!")
#
#     while True:
#         try:
#             data = socket_con.recv(2048)
#
#             datastr = data.decode()
#             if datastr == "3":
#                 print(datastr)
#                 camera1 = cv2.VideoCapture(1)
#                 camera1.set(3, 640)
#                 camera1.set(4, 480)
#                 reSt, reDis = estimation.eismation(camera1)
#                 camera1.release()
#                 sendData = {
#                     '1': 3,
#                     '2': reSt,
#                     '3': reDis
#                 }
#                 print(sendData)
#                 messagestr = json.dumps(sendData)
#                 message = bytes(messagestr, encoding="utf8")
#                 socket_con.send(message)
#             if datastr == "4":
#                 print(datastr)
#                 sendData = {
#                     '1': 4,
#                     '2': get_fingure_label()
#                 }
#                 print(sendData)
#                 messagestr = json.dumps(sendData)
#                 message = bytes(messagestr, encoding="utf8")
#                 socket_con.send(message)
#             if datastr == "5":
#                 print(datastr)
#                 # resultstr = ocrweb()
#                 sendData = {
#                     '1': 5,
#                     '2': "说文解词"
#                 }
#                 messagestr = json.dumps(sendData)
#                 message = bytes(messagestr, encoding="utf8")
#                 socket_con.send(message)
#         except:
#             print("cash one !")
#

try:
    print(get_fingure_label())
except:
    print("capture error")
