import torch
from torch.autograd import Variable

import cv2
import numpy as np
import time

from src.net_arc import Backbone
from src.identification_record import IdentificationRecord


def load_fr_model(path):
    """
    Loads the model from the path to cuda and sets to eval mode
    :return: NN for FR
    """
    net = Backbone(num_layers=50, drop_ratio=0.6, mode='ir_se').cuda()
    net.load_state_dict(torch.load(path))
    net.eval()
    return net


def extract_features(aligned_face, net):
    """
    Given an aligned face image computes 512 dimensional FR feature for the pic and horizontally flipped image
    (used for db)
    returns [2,512] dimensional tensor corresponding to the image
    :param aligned_face: face that is aligned to the key points
    :param net: NN trained for FR
    :return: [2,512] dimensional tensor corresponding to the feature
    """

    shape = (112, 112)

    imglist = [aligned_face, cv2.flip(aligned_face, 1)]
    for i in range(len(imglist)):

        imglist[i] = imglist[i][:, :, ::-1]
        imglist[i] = imglist[i].transpose(2, 0, 1).reshape((1, 3, shape[1], shape[0]))
        imglist[i] = (imglist[i] - 127.5) / 128.0

    img = np.vstack(imglist)
    with torch.no_grad():
        # noinspection PyArgumentList
        img = Variable(torch.from_numpy(img).float()).cuda()
        output = net(img)

    return output


def extract_multiple_features(aligned_face, net):
    """
    Given an aligned face image computes 512 dimensional FR feature for the pic only (used for inference)
    returns [1,512] dimensional tensor corresponding to the image
    :param aligned_face: face image that is aligned to the key points
    :param net: NN trained for FR
    :return: [1,512] dimensional tensor corresponding to the feature
    """

    aligned_face = aligned_face[:, :, ::-1]

    aligned_face = aligned_face.transpose(0, 3, 1, 2).astype('float32')
    aligned_face = (aligned_face - 127.5) / 128.0

    img = aligned_face
    with torch.no_grad():
        img = torch.from_numpy(img).cuda()
        output = net(img)
    return output


def identify_multiple_faces(test_feature, face_keys, user_records):
    """
    Given multiple test features compares against user_records and return identification records
    :param test_feature: matrix of test features of shape (n,512, 1, 1)
    :param face_keys: list of face_ids of the test features
    :param user_records: UserRecords class instance
    :return: list of identifications
    """
    if test_feature.shape[0] == 0:
        return {}, {}

    test_feature = test_feature.view(-1, 512, 1, 1)
    feature_db = user_records.feature_db.transpose(0, 1).unsqueeze(0)
    diff = test_feature - feature_db
    dist = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
    min_dist, _ = torch.min(dist, dim=1)
    minimum, min_idx = torch.min(min_dist, dim=1)

    identifications = {}
    t = time.time()
    for k, key in enumerate(face_keys):
        bb_key = int(key.split('_')[-1])
        conf = minimum[k].item()
        if conf > 2:
            conf = 2
        try:
            uid = user_records.get_uid(min_idx[k].item())
        except KeyError:
            uid = 0

        conf = np.cos(2 * np.arcsin(conf / 2))
        identifications[bb_key] = IdentificationRecord(uid, conf, t)

    return identifications

def compare_two_faces(test_feature,face_keys,features_f2):
    """
    Given multiple test features compares against user_records and return identification records
    :param test_feature: matrix of test features of shape (n,512, 1, 1)
    :param face_keys: list of face_ids of the test features
    :param user_records: UserRecords class instance
    :return: list of identifications
    """
    if test_feature.shape[0] == 0:
        return {}, {}

    test_feature = test_feature.view(-1, 512, 1, 1)
    feature_db = features_f2.transpose(0, 1).unsqueeze(0).unsqueeze(-1)
    print('shape:',feature_db.size(),test_feature.size())
    diff = test_feature - feature_db
    dist = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=1))
    min_dist, _ = torch.min(dist, dim=1)
    minimum, min_idx = torch.min(min_dist, dim=1)

    identifications = {}
    t = time.time()
    for k, key in enumerate(face_keys):
        bb_key = int(key.split('_')[-1])
        conf = minimum[k].item()
        if conf > 2:
            conf = 2

        conf = np.cos(2 * np.arcsin(conf / 2))
        identifications[str(bb_key)+'_conf'] = conf

    return identifications