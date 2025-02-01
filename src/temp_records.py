import time
from src.visitortags_api_access_functions import api_connector
from src.visualization_utils import get_display_confidence


class LocalRecords:
    """
    A class used to keep track of recent identifications, and make identification API calls only when necessary
    """

    def __init__(self, size, ignore_time):
        """
        Initializes a Local record class
        :param size: size of the local stack
        :param ignore_time: the time in between which a hit will be ignored
        """
        if size < 4:
            size = 4
        self.time_stamps = [0] * size
        self.recent_ids = [""] * size
        self.ignore_time = ignore_time

    def add_visitor_record(self, uid, conf, time_stamp, cam_id):
        """
        internal Function used to make an API call of a new record and update the local stack
        :param uid: user_id
        :param conf: confidence
        :param time_stamp: timestamp
        :param cam_id: cam_id where identification occured
        :return: None
        """

        self.time_stamps.pop(0)
        self.time_stamps.append(time_stamp)
        self.recent_ids.pop(0)
        self.recent_ids.append(uid)
        api_connector.post_visitor_identification(uid, get_display_confidence(conf), cam_id, time_stamp)

    def update(self, new_identification_records, cam_id):
        """
        External function that gets called with validated identifications on each new frame
        :param new_identification_records: new validated identification records corresponding to the frame
        :param cam_id: id of the camera which sent the frame
        :return:
        """
        current_time = time.time()
        for k, record in enumerate(new_identification_records):
            uid = record.user_id
            if uid == 0:
                continue
            else:
                try:
                    # get the latest occurance
                    idx = len(self.recent_ids) - self.recent_ids[::-1].index(uid) - 1
                except ValueError:
                    self.add_visitor_record(uid, record.confidence, record.identified_time, cam_id)
                else:
                    if (current_time - self.time_stamps[idx]) > self.ignore_time:
                        self.add_visitor_record(uid, record.confidence, record.identified_time, cam_id)
