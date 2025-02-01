import threading
import cv2

from src.visualization_utils import blank_frame
from src.visitortags_api_access_functions import api_connector


class VideoCaptureAsync:
    """
    Class for reading video frames asynchronously and to write them in a temporary buffer.
    """

    def __init__(self, url=0, cam_id=0, width=1440, height=1080):
        """
        initiates a VideoCaptureAsync object
        :param url: url/path of the camera
        :param cam_id: source id
        :param width: width of the frame
        :param height: height of the frame
        """
        self.cam_id = cam_id

        try:
            url = int(url)
        except (ValueError, TypeError):
            url = str(url)

        self.cap = cv2.VideoCapture(url)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

        try:
            self.h = self.frame.shape[0]
            self.w = self.frame.shape[1]

        except AttributeError:
            self.h = height
            self.w = width

        self.cam_id = cam_id
        self.offline_flag = False

    def set(self, var1, var2):
        """
        sets a certain attribute of the underlying cv2.VideoCapture object to a particular value
        :param var1: attribute
        :param var2: value
        :return: none
        """
        self.cap.set(var1, var2)

    def isOpened(self):
        """
        checks whether the cv2.VideoCapture object is opened (accessible)
        :return: True/False
        """
        return self.cap.isOpened()

    def start(self):
        """
        Starts the asynchronous thread corresponding to reading the frames
        :return: None
        """
        if self.started:
            print('[!] Asynchronous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()

    def update(self):
        """
        asynchronously read the latest frame and updates the buffer variable
        :return: None
        """
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        """
        when invoked returns the latest read frame and  grabbed status
        :return: grabbed status, frame
        """

        with self.read_lock:
            if self.grabbed:
                frame = self.frame.copy()

            else:
                frame = blank_frame()
                if not self.offline_flag:
                    api_connector.post_cam_offline(self.cam_id)
                    self.offline_flag = True

            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        """
        Stops asynchronous read
        :return: None
        """
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        """releases the underlying cv2.VideoCapture"""
        self.cap.release()
