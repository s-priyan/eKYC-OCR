import json
import numpy as np

from src.fr_functions import load_fr_model, extract_features, extract_multiple_features, identify_multiple_faces, compare_two_faces
from src.fd_functions import load_fd_model, detect_landmarks, align_test_faces
from src.helper_functions import validate_user_identification_single_frame
from src.config_parser import conf

min_face_size = conf.hyper_params.fd_model.min_detection_size


class FeatureExtractor:
    """
    Class to be used for extending to API for facial feature extraction. Also supports single image face recognition.
    """
    def __init__(self, path):
        """
        Initalizes the face detection and face recognition models as instance parameters.
        :param path: path of the FR model
        """
        self.fd_net = load_fd_model()
        self.fr_net = load_fr_model(path)

    def extract_feature(self, img):
        """
        Given an image detects and extracts the facial feature vector. Used for API exposure.
        :param img: image (cv2 instance)
        :return: error code and vector (if successful); error code and error message (if not)
        """
        test_landmark_list, test_bb_list = detect_landmarks(img, net_list=self.fd_net, min_face_size=min_face_size)
        if len(test_landmark_list) > 1:
            return 6, "More than one face"
        elif len(test_landmark_list) == 0:
            return 5, "No faces found"

        aligned_test_face = align_test_faces(img, test_landmark_list)
        feature = extract_features(aligned_test_face['face_0'], self.fr_net)
        feature_vector = feature.tolist()
        feature_vector = json.dumps(feature_vector, separators=(',', ':'), sort_keys=True, indent=4)
        return 0, feature_vector
        
    def extract_feature_array(self, img):
        """
        Given an image detects and extracts the facial feature vector. Used for API exposure.
        :param img: image (cv2 instance)
        :return: error code and vector (if successful); error code and error message (if not)
        """
        test_landmark_list, test_bb_list = detect_landmarks(img, net_list=self.fd_net, min_face_size=min_face_size)
        if len(test_landmark_list) > 1:
            return 6, "More than one face"
        elif len(test_landmark_list) == 0:
            return 5, "No faces found"

        aligned_test_face = align_test_faces(img, test_landmark_list)
        feature = extract_features(aligned_test_face['face_0'], self.fr_net)
        return 0, feature

    def recognize_faces(self, img, user_records):
        """
        Given an image as input find matching faces against the input user_records object
        :param img: input image (cv2 instance)
        :param user_records: UserRecord instance against which the faces are to be compared
        :return: (if success)error code and, list of user_ids and confidence; (else) error code and error message
        """
        test_landmark_list, test_bb_list = detect_landmarks(img, net_list=self.fd_net, min_face_size=min_face_size)
        if len(test_landmark_list) == 0:
            return 5, "No faces found"

        aligned_test_faces = align_test_faces(img, test_landmark_list)
        inp_faces = np.stack(list(aligned_test_faces.values()))
        face_keys = list(aligned_test_faces.keys())
        test_features = extract_multiple_features(inp_faces, self.fr_net)

        identifications = identify_multiple_faces(test_features, face_keys, user_records)
        validated_identifications = validate_user_identification_single_frame(identifications)
        if len(validated_identifications) == 0:
            return 6, "user not recognized"
        return_records = []
        for record in validated_identifications:
            return_records.append({"user_id": record.user_id, "confidence": record.confidence})

        return 0, return_records
        
    def compare_faces(self, img_w,img_id):
        """
        Given an images as input compare faces
        :param img: input image (cv2 instance)
        :param user_records: UserRecord instance against which the faces are to be compared
        :return: (if success)error code and, list of user_ids and confidence; (else) error code and error message
        """
        test_landmark_list, test_bb_list = detect_landmarks(img_w, net_list=self.fd_net, min_face_size=min_face_size)
        if len(test_landmark_list) == 0:
            return 5, "No faces found"

        aligned_test_faces = align_test_faces(img_w, test_landmark_list)
        inp_faces = np.stack(list(aligned_test_faces.values()))
        face_keys = list(aligned_test_faces.keys())
        test_features = extract_multiple_features(inp_faces, self.fr_net)

        identifications = compare_two_faces(test_features,face_keys,img_id)
        #validated_identifications = validate_user_identification_single_frame(identifications)
        #if len(validated_identifications) == 0:
         #   return 6, "user not recognized"
        #return_records = []
        #for record in validated_identifications:
         #   return_records.append({"user_id": record.user_id, "confidence": record.confidence})

        return 0, identifications
    