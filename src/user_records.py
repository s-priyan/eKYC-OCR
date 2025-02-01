from collections import OrderedDict
import torch
import json
import numpy as np


class UserRecords:
    """
    Class to hold the user database (face vectors, user_ids)
    """
    def __init__(self, max_db_size):
        """
        Initializes a UserRecord class
        :param max_db_size: maximum size of the database
        """
        self.feature_db_idx = OrderedDict()
        self.user_ids = OrderedDict()
        self.next_user_idx = 0
        self.feature_db = torch.ones(2, 512, max_db_size).cuda()

    def add(self, uid, feature_vector):
        """
        adds a new record to the database
        :param uid: user_id
        :param feature_vector: feature vector as text
        :return: status True or False
        """
        try:
            idx = self.feature_db_idx[uid]
        except KeyError:
            self.feature_db_idx[uid] = self.next_user_idx
            self.user_ids[self.next_user_idx] = uid
            vector = json.loads(feature_vector)
            vector = np.array(vector)
            vector = torch.from_numpy(vector).cuda()
            self.feature_db[:, :, self.next_user_idx] = vector
            self.next_user_idx += 1
            return True
        else:
            return False

    def update(self, uid, feature_vector):
        """
        updates an existing record in the database
        :param uid: user_id
        :param feature_vector: feature vector as text
        :return: status True or False
        """

        try:
            idx = self.feature_db_idx[uid]
        except KeyError:
            return False
        vector = json.loads(feature_vector)
        vector = np.array(vector)
        vector = torch.from_numpy(vector).cuda()
        self.feature_db[:, :, idx] = vector
        return True

    def delete(self, uid):
        """
        deletes an existing record
        :param uid: ser_id to delete
        :return: status True or False
        """
        try:
            feature_idx = self.feature_db_idx[uid]
        except KeyError:
            return False
        self.feature_db[:, :, feature_idx] = torch.ones(2, 512).cuda()
        del self.user_ids[feature_idx]
        del self.feature_db_idx[uid]
        return True

    def get_uid(self, feature_idx):
        """
        given the index of the feature returns the user_id
        :param feature_idx: index of the feature in the feature matrix
        :return: user_id
        """
        return self.user_ids[feature_idx]
