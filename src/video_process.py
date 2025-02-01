import threading

from src.visualization_utils import blank_frame

from src.feature_tracker import FeatureTracker
from src.fr_functions import load_fr_model
from src.fd_functions import load_fd_model

from src.temp_records import LocalRecords
from src.helper_functions import detect_faces_in_frame_optimized_ft
from src.config_parser import conf

path = conf.hyper_params.fr_model.path

tracker_min_appeared = conf.hyper_params.feature_tracker.min_frames_appeared_to_validate
tracker_max_disappeared = conf.hyper_params.feature_tracker.max_frames_disappeared_to_discard
tracker_window = conf.hyper_params.feature_tracker.validation_window_size

local_record_ignore_time = conf.hyper_params.local_record.rerecord_ignore_time_seconds
local_record_stack_size = conf.hyper_params.local_record.stack_size


class VideoProcessorAsync:
    """
    Class for processing video frames asynchronously
    """
    def __init__(self, video_getter, user_db):
        """
        Initiates a VideoProcessorAsync class
        :param video_getter: VideoCaptureAsync instance
        :param user_db: UserRecords instance
        """

        self.frame = None
        self.started = False
        self.thread = None
        self.read_lock = threading.Lock()

        self.video_getter = video_getter
        self.cam_id = video_getter.cam_id

        self.ft = FeatureTracker(min_appeared=tracker_min_appeared, max_disappeared=tracker_max_disappeared,
                                 window_size=tracker_window)
        self.fr_net = load_fr_model(path)
        self.fd_net_list = load_fd_model()
        self.local_records = LocalRecords(local_record_stack_size, local_record_ignore_time)

        self.user_db = user_db

    def start(self):
        """
        Starts the asynchronous thread corresponding to processing the frames
        :return: None
        """
        if self.started:
            print('[!] Asynchronous video processing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.process, args=())
        self.thread.start()

    def process(self):
        """
        asynchronously processes the latest frame and updates the local records
        :return: None
        """

        while self.started:
            rval, frame = self.video_getter.read()

            return_records, unknown_count = detect_faces_in_frame_optimized_ft(frame, self.ft, self.fr_net,
                                                                               self.fd_net_list, self.user_db)

            self.local_records.update(return_records, self.cam_id)

    def stop(self):
        """
        Stops asynchronous process
        :return: None
        """
        self.started = False
        self.thread.join()
