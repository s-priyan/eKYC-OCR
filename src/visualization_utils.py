import cv2
import numpy as np


def blank_frame():
    """
    Returns a blank frame of FHD resolution
    :return: cv2 instance of black frame
    """
    frame = np.zeros((1080, 1920, 3), np.uint8)
    cv2.putText(frame, 'No camera', (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
    return frame


def get_display_confidence(cos_value):
    """
    Converts the confidence into a display friendly number, using a piece wise linear mapping
    :param cos_value: cos value of the angle between the two vectors
    :return: confidence value as a float
    """
    saturation_point = 0.75
    thresh = 0.55
    guess_thresh = 0.45

    if cos_value > saturation_point:
        confidence = 1.0
    elif cos_value > thresh:
        confidence = (1.0-0.75) * (cos_value - thresh)/ (saturation_point - thresh) + 0.75
    elif cos_value > guess_thresh:
        confidence = (0.75-0.5) * (cos_value - guess_thresh)/(thresh-guess_thresh) + 0.5
    else:
        confidence = 0.0

    return confidence