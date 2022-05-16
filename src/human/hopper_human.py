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
        self.p = Process(target=self.query_preference, args=(self.query_queue, self.answer_queue))

    def query_preference(self, query_queue, answer_queue):
        serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            connectionSocket, addr = serverSocket.accept()
            try:
                message = connectionSocket.recv(1024)
                print(message)
                connectionSocket.close()
            except IOError:
                # Send response message for file not found
                connectionSocket.send('404 Not Found')
                # Close client socket
                connectionSocket.close()

    @staticmethod
    def calc_coords(frame):
        state = np.logical_and(frame[0, :, :] > 0.5, frame[1, :, :] < 0.5)
        x, y = np.argwhere(state).mean(0)
        return x, y
