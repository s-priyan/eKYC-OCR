import operator
import time
import torch
import numpy as np

from src.config_parser import conf
from src.fd_functions import detect_landmarks, align_test_faces
from src.fr_functions import identify_multiple_faces, extract_multiple_features


fr_threshold = conf.hyper_params.fr_model.threshold
fr_guess_threshold = conf.hyper_params.fr_model.guess_threshold

DEBUG = conf.debug_params.run_time_print
min_face_size = conf.hyper_params.fd_model.min_detection_size


def validate_user_identification_single_frame(identifications):
    """
    When a single frame/ image based identification needs to be validated. This function can be used.
    :param identifications: list of Identification objects
    :return: list of Identification objects
    """
    validated_identifications = []

    for i in identifications:
        ident = identifications[i]
        if ident.confidence > fr_threshold:
            validated_identifications.append(ident)
    return validated_identifications


def validate_user_moving_window(id_list, conf_list, val_id, top_conf, window_size):
    """
    When a user face is tracked for more than n frames this function is invoked to assign a name. Assigns the name (id)
    that appears with maximum confidence to the tracker ID
    :param id_list: list of accumulated top confidence ids
    :param conf_list: list of confidence corresponding to each id in id list
    :param val_id: previously validated id
    :param top_conf: previously registered highest confidence
    :param window_size: size of the moving window
    :return: visual record object which contains the bbox coordinates
    """

    threshold = fr_threshold
    guess_threshold = fr_guess_threshold

    if len(id_list) > window_size:
        id_list.pop(0)
        conf_list.pop(0)

    id_dict = {}
    for i in range(len(id_list)):
        current_id = id_list[i]
        if current_id in id_dict.keys():
            id_dict[current_id] += conf_list[i]
        else:
            id_dict[current_id] = conf_list[i]

    max_id = max(id_dict.items(), key=operator.itemgetter(1))[0]
    confidence = id_dict[max_id] / len(id_list)

    if confidence < guess_threshold:
        known = 0
    elif confidence < threshold:
        known = 1
    elif id_list.count(max_id) < len(id_list)/2:
        known = 1
    else:
        known = 2
        if max_id != val_id:
            val_id = max_id
            top_conf = confidence
        elif confidence > top_conf:
            top_conf = confidence

    return val_id, top_conf


def detect_faces_in_frame_optimized_ft(frame, ft, fr_net, fd_net_list, user_records):
    """
    Wrapper function to detect, align, extract features, identify, and track faces in a given frame.
    :param frame: cv2 instance. (frame of a video stream)
    :param ft: feature tracker object instance
    :param fr_net: face recognition network
    :param fd_net_list: face detection network tuple
    :param user_records: UserRecords object
    :return: validated identification records and (cumulative) unknown count
    """
    
    t = time.time()
    test_landmark_list, test_bb_list = detect_landmarks(frame, net_list=fd_net_list, min_face_size=min_face_size)
    if DEBUG:
        print("Detect---------", time.time() - t)
    t = time.time()
    aligned_test_faces = align_test_faces(frame, test_landmark_list)
    if DEBUG:
        print("Align---------", time.time() - t)
    t = time.time()
    if len(aligned_test_faces) == 0:
        test_features = torch.zeros(0, 512)
        face_keys = []
    else:
        inp_faces = np.stack(list(aligned_test_faces.values()))
        face_keys = list(aligned_test_faces.keys())
        test_features = extract_multiple_features(inp_faces, fr_net)
    if DEBUG:
        print("Extract features---------", time.time() - t)
    t = time.time()
    identifications = identify_multiple_faces(test_features, face_keys, user_records)
    if DEBUG:
        print("Recognize---------", time.time() - t)
    t = time.time()
    uid_records, unknown_count = ft.update(test_bb_list, test_features, identifications)
    if DEBUG:
        print("update---------", time.time() - t)

    return uid_records, unknown_count
