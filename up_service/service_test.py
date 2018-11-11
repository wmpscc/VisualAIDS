import socket
import time
import sys
import json

import base64



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


print("start socket: TCP...")
socket_tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

HOST_IP = getip()
HOST_PORT = 7653

print("tcp server listen @ %s:%d!" % (HOST_IP, HOST_PORT))
host_addr = (HOST_IP, HOST_PORT)
socket_tcp.bind(host_addr)
socket_tcp.listen(1)

while True:
    print("waiting for connection...")
    socket_con, (client_ip, client_port) = socket_tcp.accept()
    print("connectionaccepted from %s." % client_ip)

    # socket_con.send(b"welcome!")

    while True:
        data = socket_con.recv(1024)
        if not data:
            time.sleep(0.5)
            continue
        datastr = data.decode()
        if datastr == "3":
            print(datastr)
            sendData = {
                '1': 3,
                '2': "桌子",
                '3': "椅子"
            }
            messagestr = json.dumps(sendData)
            message = bytes(messagestr, encoding="utf8")
            socket_con.send(message)
        if datastr == "4":
            print(datastr)
            sendData = {
                '1': 4,
                '2': "汽车"
            }
            messagestr = json.dumps(sendData)
            message = bytes(messagestr, encoding="utf8")
            socket_con.send(message)
        if datastr == "5":
            print(datastr)
            # resultstr = ocrweb()
            sendData = {
                '1': 5,
                '2': "杯子"
            }
            messagestr = json.dumps(sendData)
            message = bytes(messagestr, encoding="utf8")
            socket_con.send(message)
        # if datastr == "9":
        #     socket_con.close()
        #     print("tcp_close")
    # socket_con.close()
    # print("tcp_close")