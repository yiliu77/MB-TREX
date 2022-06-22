from .human import Human
from multiprocessing import Process, Queue
import base64
import time
import io
from multiprocessing import Process, Queue
import socket
import os
from PIL import Image as im
import numpy as np

def generate_image_encoding(images):
    all_encoded_images = ""
    for i in range(len(images)):
        buf = io.BytesIO()
        im_resize = im.fromarray(images[i]).resize((500, 500))
        im_resize.save(buf, format='JPEG')
        encoded_string = base64.b64encode(buf.getvalue()).decode('utf-8')
        all_encoded_images += '<img style="position:absolute;" src="data:image/png;base64, {}">'.format(encoded_string)
    return all_encoded_images

class HopperHuman(Human):

    def __init__(self, env, same_margin, actual_human):
        self.env = env
        self.same_margin = same_margin
        if actual_human:
            self.query_preference = self.query_preference_real
            self.query_queue = Queue()
            self.answer_queue = Queue()
            self.p = Process(target=self.start_server, args=(self.query_queue, self.answer_queue))
            self.p.start()
        else:
            self.query_preference = self.query_preference_artificial
            self.same_margin = same_margin



    def query_preference_artificial(self, traj1, traj2):
        states1 = traj1[:, :11]
        states2 = traj2[:, :11]
        acs1 = traj1[:, 11:]
        acs2 = traj2[:, 11:]
        cost1 = self.env.get_expert_cost(states1, acs1).sum()
        cost2 = self.env.get_expert_cost(states2, acs2).sum()
        if abs(cost1 - cost2) > self.same_margin:
            label = int(cost1 > cost2)
        else:
            label = 0.5
        return label

    def query_preference_real(self, traj1, traj2):
        traj1 = self.convert_to_images(traj1)
        traj2 = self.convert_to_images(traj2)
        self.query_queue.put((traj1, traj2, False))
        return self.answer_queue.get()

    def convert_to_images(self, trajectory):
        images = []
        for i in range(len(trajectory)):
            self.env.reset(pos=trajectory[i, :11])
            images.append(self.env._get_obs(use_images=True))
        return np.array(images)

    def start_server(self, query_queue, answer_queue):
        last_image1, last_image2, last_validate = None, None, None
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serverSocket.bind(("localhost", 8890))
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
                <body><h1> Currently no queries. Please wait </body> </html>""".format(query_queue.qsize()).encode(
                    'utf-8'))
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
                connectionSocket.sendall(
                    lines.replace("{images1}", generate_image_encoding(images1)).replace("{images2}",
                                                                                         generate_image_encoding(
                                                                                             images2)).
                    replace("{validate}", "Validate: " + str(last_validate)).encode('utf-8'))
                connectionSocket.close()