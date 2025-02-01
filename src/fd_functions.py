from collections import OrderedDict

import cv2
import numpy as np

from src.detector import detect_faces, load_models
from src.matlab_cp2tform import get_similarity_transform_for_cv2


def alignment(src_img, src_pts):
    """
    Alignment function. Makes use of Affine transform. The ref point used in this are the ref points used by the arcface
    team while training
    :param src_img: image to be aligned
    :param src_pts: facial landmarks in src image
    :return: aligned cropped image
    """

    ref_pts = [
        [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]
    ]
    crop_size = (112, 112)

    src_pts = np.array(src_pts).reshape(5, 2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img


def load_fd_model():
    """
    Loads the face detection models and returns as a tuple
    :return: list containing pnet rnet and onet
    """

    return load_models()


def detect_landmarks(img, net_list, min_face_size=60.0):
    """
    Given an image detects faces in the image and returns corresponding facial landmarks and bounding boxes
    :param img: cv2 instance
    :param net_list: tuple of networks pnet, rnet, and onet
    :return: dictionary of landmarks and a list of bounding boxes
    """

    test_landmark_list = OrderedDict()
    img = img[:, :, ::-1]

    bounding_boxes, test_landmarks = detect_faces(img, pnet=net_list[0], rnet=net_list[1], onet=net_list[2],
                                                  min_face_size=min_face_size, thresholds=[0.6, 0.7, 0.85])
    for face in range(len(test_landmarks)):
        i = test_landmarks[face]
        landmark_reshaped = [i[0], i[5], i[1], i[6], i[2], i[7], i[3], i[8], i[4], i[9]]
        landmark_reshaped = np.around(landmark_reshaped).astype(int)
        a = 'face_' + str(face)
        test_landmark_list[a] = landmark_reshaped
    bounding_boxes_list = bounding_boxes

    return test_landmark_list, bounding_boxes_list


def align_test_faces(img, test_landmark_list):
    """
    Given an image with multiple faces and detected landmarks returns the aligned faces as a dictionary
    :param img: image with multiple faces
    :param test_landmark_list: dictionary of landmarks in the image
    :return: dictionary of aligned faces
    """

    aligned_images = OrderedDict()
    for key in test_landmark_list:
        aligned_images[key] = alignment(img, test_landmark_list[key])
    return aligned_images
