import numpy as np
from human.simple_point_human import Human
import base64
import socket
import io
from PIL import Image as im
import os
import time
from multiprocessing import Process, Queue


def generate_image_encoding(images):
    all_encoded_images = ""
    for i in range(len(images)):
        buf = io.BytesIO()
        im_resize = im.fromarray(np.transpose(255 * images[i], (1, 2, 0)).astype(np.uint8)).resize((500, 500))
        im_resize.save(buf, format='JPEG')
        encoded_string = base64.b64encode(buf.getvalue()).decode('utf-8')
        all_encoded_images += '<img style="position:absolute;" src="data:image/png;base64, {}">'.format(encoded_string)
    return all_encoded_images


class HopperHuman(Human):
    def __init__(self, env):
        self.env = env

        self.query_queue = Queue()
        self.answer_queue = Queue()
        self.p = Process(target=self.start_server, args=(self.query_queue, self.answer_queue))
        self.p.start()

    def query_preference(self, paired_states1, paired_states2, validate=False):
        self.query_queue.put((paired_states1, paired_states2, validate))
        return self.answer_queue.get()

    def start_server(self, query_queue, answer_queue):
        last_image1, last_image2, last_validate = None, None, None
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serverSocket.bind(("localhost", 8892))
        serverSocket.listen(1)
        while True:
            connectionSocket, addr = serverSocket.accept()
            message = connectionSocket.recv(10240)
            if "Pref1" in str(message):
                answer_queue.put(0)
            if "Pref2" in str(message):
                answer_queue.put(1)
            if "Pref" in str(message):
                last_image1 = None
                last_image2 = None
                last_validate = None
                time.sleep(0.1)

            if query_queue.empty() and last_image1 is None:
                connectionSocket.sendall(b'HTTP/1.1 200 OK\r\n\r\n')
                connectionSocket.sendall("""<html><head> <meta http-equiv="refresh" content="1" /></head>
                <body><h1> Currently no queries. Please wait </body> </html>""".format(query_queue.qsize()).encode('utf-8'))
                connectionSocket.close()
            else:
                if last_image1 is None:
                    images1, images2, validate = query_queue.get()
                    last_image1 = images1
                    last_image2 = images2
                    last_validate = validate
                else:
                    images1, images2 = last_image1, last_image2
                dirname = os.path.dirname(__file__)
                filename = os.path.join(dirname, 'files/preference.html')
                with open(filename) as f:
                    lines = "".join(f.readlines())

                connectionSocket.sendall(b'HTTP/1.1 200 OK\r\nContent-Type: text/html;charset=utf-8\r\n\r\n')
                connectionSocket.sendall(lines.replace("{images1}", generate_image_encoding(images1)).replace("{images2}", generate_image_encoding(images2)).
                                         replace("{validate}", "Validate: " + str(last_validate)).encode('utf-8'))
                connectionSocket.close()
