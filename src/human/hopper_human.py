import numpy as np
from human.simple_point_human import Human
import wandb
import socket
from multiprocessing import Process, Queue


class HopperHuman(Human):
    def __init__(self, env):
        self.env = env
        self.query_queue = Queue()
        self.answer_queue = Queue()
        self.p = Process(target=self.start_server, args=(self.query_queue, self.answer_queue))
        self.p.start()

    def query_preference(self, paired_states1, paired_states2, validate=False):
        self.query_queue.put((paired_states1, paired_states2))
        return self.answer_queue.get()

    def start_server(self, query_queue, answer_queue):
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serverSocket.bind(("localhost", 8893))
        serverSocket.listen(1)
        while True:
            connectionSocket, addr = serverSocket.accept()
            message = connectionSocket.recv(10240)
            print(message)
            if query_queue.empty():
                connectionSocket.sendall(b'HTTP/1.1 200 OK\r\n\r\n')
                connectionSocket.sendall("""<html>
                       < body >
                       < h1 > Hello World < / h1 > this is my server! {} < / body > < / html >""".format(query_queue.qsize()).encode('utf-8'))
                connectionSocket.close()
            else:
                connectionSocket.sendall(b'HTTP/1.1 200 OK\r\n\r\n')
                connectionSocket.sendall("""<html>
                       < body >
                       < h1 > Hello World < / h1 > this is my server! {} < / body > < / html >""".format(query_queue.qsize()).encode('utf-8'))
                connectionSocket.close()
