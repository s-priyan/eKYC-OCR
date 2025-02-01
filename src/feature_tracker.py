from collections import OrderedDict
import time
import numpy as np
import torch

from src.helper_functions import validate_user_moving_window
from src.identification_record import IdentificationRecord


class FeatureTracker:
    """
    Class to track faces in multiple frames and assign them unique object_id.
    This is useful for multi frame validation.
    """
    def __init__(self, threshold=0.9, max_disappeared=10, min_appeared=30, window_size=10):
        """
        Initializes a tracker instance.
        :param threshold: cos threshold value to assign two faces same object_id
        :param max_disappeared: maximum number of frames an object can remain undetected, without being discarded.
        :param min_appeared: minimum number of frames an object_id should appear to be validated.
        :param window_size: size of the validation window.
        """

        self.next_object_id = 0
        self.centroids = OrderedDict()
        self.features = OrderedDict()
        self.disappeared = OrderedDict()
        self.appeared = OrderedDict()
        self.user_id_list = OrderedDict()
        self.conf_list = OrderedDict()

        self.known = OrderedDict()
        self.valid_user_id = OrderedDict()
        self.top_conf = OrderedDict()

        self.max_disappeared = max_disappeared
        self.min_appeared = min_appeared
        self.window_size = window_size
        self.unknown_count = 0
        self.threshold = threshold

    def register(self, centroid, feature, user_id, confidence):
        """
        internal function used to register a new trackable object.
        :param centroid: center of the face rectangle
        :param feature: feature vector
        :param user_id: maximum likely user_id
        :param confidence: confidence of that user_id
        :return: None
        """

        self.centroids[self.next_object_id] = centroid
        self.features[self.next_object_id] = feature
        self.disappeared[self.next_object_id] = 0
        self.appeared[self.next_object_id] = 1

        self.user_id_list[self.next_object_id] = [user_id]
        self.conf_list[self.next_object_id] = [confidence]

        self.known[self.next_object_id] = False
        self.valid_user_id[self.next_object_id] = 0
        self.top_conf[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Internal function used to discard an object_id. This is used when an object has gone out of FOV.
        :param object_id: the object id to be removed
        :return: None
        """
        if (not self.known[object_id]) and self.appeared[object_id] > 30:
            self.unknown_count += 1

        del self.centroids[object_id]
        del self.features[object_id]
        del self.disappeared[object_id]
        del self.appeared[object_id]
        del self.user_id_list[object_id]
        del self.conf_list[object_id]

        del self.known[object_id]
        del self.valid_user_id[object_id]
        del self.top_conf[object_id]

    def update(self, rects, input_features, input_identifications):
        """
        In every frame this function will be called to update the tracker. And if any positive hits are present,
        it will be returned.
        :param rects: detected face rectangle coordinates as tuple
        :param input_features: features corresponding to the faces
        :param input_identifications: list of IdentificationRecord objects that are recognized
        :return: list of IdentificationRecord objects that are validated and unknown count.
        """
        validated_identifications = []
        self.unknown_count = 0

        if len(rects) == 0:

            disappeared_copy = self.disappeared.copy()
            for object_id in disappeared_copy.keys():
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] >= self.max_disappeared:
                    self.deregister(object_id)

            return validated_identifications, self.unknown_count

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        input_rects = np.zeros((len(rects), 4), dtype="int")
        for (i, (start_x, start_y, end_x, end_y, _)) in enumerate(rects):
            c_x = int((start_x + end_x) / 2.0)
            c_y = int((start_y + end_y) / 2.0)
            input_centroids[i] = (c_x, c_y)

            input_rects[i] = (int(start_x), int(start_y), int(end_x), int(end_y))

        if len(self.centroids) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i], input_features[i], input_identifications[i].user_id,
                              input_identifications[i].confidence)

        else:

            object_ids = list(self.features.keys())
            object_features = torch.stack(list(self.features.values()))
            diff = object_features.unsqueeze(-1) - input_features.transpose(0, 1).unsqueeze(0)
            dist = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
            min_dist, min_idx = torch.min(dist, dim=1)

            rows = min_dist.argsort().cpu().numpy()

            cols = min_idx[rows].cpu().numpy()

            used_rows = set()
            used_cols = set()
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if dist[row, col] > self.threshold:
                    continue

                object_id = object_ids[row]
                self.centroids[object_id] = input_centroids[col]
                self.features[object_id] = input_features[col]
                self.disappeared[object_id] = 0
                self.appeared[object_id] += 1

                user_id = input_identifications[col].user_id
                conf = input_identifications[col].confidence
                self.user_id_list[object_id].append(user_id)
                self.conf_list[object_id].append(conf)
                if self.appeared[object_id] > self.min_appeared:
                    validated_id, top_conf = validate_user_moving_window(self.user_id_list[object_id],
                                                                         self.conf_list[object_id],
                                                                         self.valid_user_id[object_id],
                                                                         self.top_conf[object_id], self.window_size)
                    self.top_conf[object_id] = top_conf
                    self.valid_user_id[object_id] = validated_id

                    validated_identifications.append(IdentificationRecord(validated_id, top_conf, time.time()))

                    if validated_id != 0:
                        self.known[object_id] = True

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, dist.shape[0])).difference(used_rows)
            unused_cols = set(range(0, dist.shape[1])).difference(used_cols)

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1

                if self.disappeared[object_id] >= self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register(input_centroids[col], input_features[col], input_identifications[col].user_id,
                              input_identifications[col].confidence)

        return validated_identifications, self.unknown_count
