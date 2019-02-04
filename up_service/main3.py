import threading
import json
import socket
import cv2
import numpy as np
import movidus_utils
import yolo_utils
from graph_ace import ace_algorithm
from fingure_detection import fingure_top
from Object_Detection import object
from FinalCode import DisparityEstimation


class Server(object):

    def __init__(self, host, port):
        self.host = host  # ip地址
        self.port = port  # 端口号
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建一个套接字，采用tcp连接
        # 设置socket，第一个参数使用正在使用的socket选项，第二个参数代表当socket关闭后，本地端用
        # 于该socket的端口号立刻就可以被重用。通常来说，只有经过系统定义一段时间后，才能被重用。
        # 1表示将SO_REUSEADDR标记为TRUE
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))  # 将端口号绑定到IP地址
        self.ace = ace_algorithm()
        self.fd = fingure_top()
        self.obd = object()
        self.estimation = DisparityEstimation()

    def get_fingure_label(self,
                          graph_file="graph/tiny-yolo-voc-1c.graph",
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
                ftop = self.fd.get_fingure_top(roi)
                if ftop != -1:
                    ftop = list(ftop)
                    ftop[0] = ftop[0] + left
                    ftop[1] = ftop[1] + top
                    ftop = tuple(ftop)
                    input_fifo.destroy()
                    output_fifo.destroy()
                    graph.destroy()
                    # dev.close()
                    # dev.destroy()
                    return self.obd.calc_target(ftop, frame_orig, dev)
            # cv2.imshow("out", frame_orig)
            k = cv2.waitKey(1) & 0xff

        cap.release()

    def listen(self):
        self.sock.listen(1)
        while True:
            client, address = self.sock.accept()  # 接受客户端的连接请求
            client.settimeout(60)  # 如果重连60秒后还没有连上则断开连接
            threading.Thread(target=self.listenToClient, args=(client, address)).start()

    def listenToClient(self, client, address):
        size = 1024  # 一次接收最大字节数
        while True:
            try:
                data = client.recv(size)
                datastr = data.decode()  # 解码为字符串
                if datastr == "3":
                    print(datastr)
                    camera1 = cv2.VideoCapture(0)
                    camera1.set(3, 640)
                    camera1.set(4, 240)
                    reSt, reDis = self.estimation.eismation(camera1)
                    camera1.release()
                    sendData = {
                        '1': 3,
                        '2': reSt,
                        '3': reDis
                    }
                    print(sendData)
                    messagestr = json.dumps(sendData)
                    message = bytes(messagestr, encoding="utf8")  # 将字符串转换为字节流
                    client.send(message)  # 发送字节流
                elif datastr == "4":
                    print(datastr)
                    name = self.get_fingure_label()
                    sendData = {
                        '1': 4,
                        '2': name if name == "发现物品" else "这是" + name
                    }
                    print(sendData)
                    messagestr = json.dumps(sendData)
                    message = bytes(messagestr, encoding="utf8")
                    client.send(message)
                elif datastr == "5":
                    print(datastr)
                    sendData = {
                        '1': 5,
                        '2': "眼镜"
                    }
                    print(sendData)
                    messagestr = json.dumps(sendData)
                    message = bytes(messagestr, encoding="utf8")
                    client.send(message)
                else:
                    # raise socket.error('Client disconnected')
                    pass
            except:
                # client.close()
                # return False
                print("crash one!")
                pass


def getip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("www.baidu.com", 0))
        ip = s.getsockname()[0]
        print(ip)
    except:
        ip = "x.x.x.x"
    finally:
        s.close()
    return ip


if __name__ == "__main__":
    while True:
        port_num = 7653  # 约定的端口号
        try:
            port_num = int(port_num)
            break
        except ValueError:
            pass
    Server(getip(), port_num).listen()
