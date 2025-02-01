import base64
import sys
import time
import binascii
import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request
from paddleocr import PaddleOCR

from src.config_parser import conf
from src.feature_extractor import FeatureExtractor
from src.my_logging import Logger
from src.video_capture import VideoCaptureAsync
from src.visitortags_api_access_functions import api_connector
from src.tenent import Tenant
# ----------------------------------------------------------------

app = Flask(__name__)

DEBUG = conf.debug_params.api_access_print
sys.stdout = Logger("output/logs/" + time.strftime('%Y-%m-%d %H:%M:%S') + ".txt")

fr_path = conf.hyper_params.fr_model.path

visitor_db = api_connector.load_visitor_db()
camera_db = api_connector.load_cam_db()

tenant = Tenant(1, camera_db, visitor_db)
tenant.initiate()

feature_extractor = FeatureExtractor(fr_path)

"""
initialize the paddleocr engine with the appropriae detction , recognition , classification models

"""
ocr = PaddleOCR(use_angle_cls=True, lang='en',use_gpu=True,  
                cls_model_dir='/home/ubuntu/.paddleocr/2.2.0.2/ocr/cls/ch_ppocr_mobile_v2.0_cls_infer' , 
                det_model_dir='/home/ubuntu/.paddleocr/2.2.0.2/ocr/det/en/en_ppocr_server_v2.0_det_infer' , 
                rec_model_dir='/home/ubuntu/.paddleocr/2.2.0.2/ocr/rec/en/en_number_server_v2.0_rec_infer') # need to run only once to download and load model into memory 

@app.route("/validate-image", methods=['POST'])
def extract_face_vector_img():
    """
    API function used to extract the feature vector corresponding to a face passed in as base64 encoded string
    in the body of the request. Returns the vector if success. Returns error code + error message if failed
    :return: error code and the vector, in case of an error returns error code and error message
    """

    if request.method != "POST":
        return jsonify(code=1, response="Invalid request type"), 400

    data = request.get_json()
    if DEBUG:
        print(data)
    try:
        img_str = data["img"]
    except KeyError:
        return jsonify(code=2, response="Invalid data keys"), 400

    try:
        img_str = bytes(img_str, 'utf-8')
        img_str = base64.b64decode(img_str)
        np_array = np.fromstring(img_str, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except (IOError, cv2.error, binascii.Error):
        return jsonify(code=3, response="Invalid image"), 400
    if (image.shape[0] < 100) or (image.shape[1] < 100):
        return jsonify(code=4, response="Image too small"), 400

    code, result = feature_extractor.extract_feature(image)
    print("result")
    print(len(list(result)))
    print(len(list(result)[0]))
    if code == 0:
        if DEBUG:
            print(result)
        return jsonify(code=0, response="Success", vector=result)
    else:
        return jsonify(code=code, response=result), 400


@app.route("/validate-camera", methods=['POST'])
def validate_camera():
    """
    API function to check if an iput camera is working before adding it to the database.
    The credentials, IP, port, and camera ID should be passed in the body of the request (JSON format)
    :return: returns the URL of the camera if accessible. return error message otherwise
    """
    if request.method == "POST":
        data = request.get_json()

        try:
            camera_id = data["camera_id"]
            camera_ip = data["camera_ip"]
            camera_port = data["rtsp_port"]
            username = data["username"]
            password = data["password"]

        except KeyError:
            return jsonify(code=2, response="Invalid data keys"), 400

        try:
            suffix = data["url_suffix"]
        except KeyError:
            camera_url = "rtsp://{}:{}@{}:{}".format(username, password, camera_ip, camera_port)
        else:
            camera_url = "rtsp://{}:{}@{}:{}".format(username, password, camera_ip, camera_port) + "/" + suffix

        if DEBUG:
            print(camera_url)
        cam_instance = VideoCaptureAsync(camera_url, camera_id)

        if cam_instance.isOpened():
            del cam_instance
            return jsonify(code=0, url=camera_url, response="Success")

        else:
            del cam_instance
            print("camera ", camera_id, " is not accessible. Please check IP and credentials")
            return jsonify(code=3, response="Unable to access camera"), 400

    else:
        return jsonify(code=1, response="Invalid request type"), 400


@app.route("/recognize-image", methods=['POST'])
def recognize_image():
    """
    API function to run face recognition on a single image (web cam captured). The image should be passed in as
    base64 encoded string in the body of the request (JSON format). The image is compared against the inmemory vectors
    and if a match is found (above the threshold) the ID and confidence is returned.
    :return: user_id and confidence level
    """
    if request.method != "POST":
        return jsonify(code=1, response="Invalid request type"), 400

    data = request.get_json()

    try:
        img_str = data["img"]
    except KeyError:
        return jsonify(code=2, response="Invalid data keys"), 400
    try:
        img_str = bytes(img_str, 'utf-8')
        img_str = base64.b64decode(img_str)
        np_array = np.fromstring(img_str, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except (IOError, cv2.error, binascii.Error):
        return jsonify(code=3, response="Invalid image"), 400
    if (image.shape[0] < 100) or (image.shape[1] < 100):
        return jsonify(code=4, response="Image too small"), 400

    code, result = feature_extractor.recognize_faces(image, tenant.visitor_db)

    if code == 0:
        if DEBUG:
            print(result)
        return jsonify(code=0, response="Success", identifications=result)
    else:
        return jsonify(code=code, response=result), 400

@app.route("/compare_image", methods=['POST'])
def compare_image():
    """
    API function to run comparison in two images (web cam captured and id). The images should be passed in as
    base64 encoded string in the body of the request (JSON format). The image is compared against the each other
    and confidence is returned.
    :return: confidence level
    """
    if request.method != "POST":
        return jsonify(code=1, response="Invalid request type"), 400

    data = request.get_json()

    try:
        img_webc = data["imgw"]
        img_id = data["imgID"]
    except KeyError:
        return jsonify(code=2, response="Invalid data keys"), 400
    try:
        img_webc = bytes(img_webc, 'utf-8')
        img_webc = base64.b64decode(img_webc)
        np_array_w = np.fromstring(img_webc, np.uint8)
        image_w = cv2.imdecode(np_array_w, cv2.IMREAD_COLOR)

        img_id = bytes(img_id, 'utf-8')
        img_id = base64.b64decode(img_id)
        np_array_id = np.fromstring(img_id, np.uint8)
        image_id = cv2.imdecode(np_array_id, cv2.IMREAD_COLOR)

    except (IOError, cv2.error, binascii.Error):
        return jsonify(code=3, response="Invalid image"), 400
    if (image_w.shape[0] < 100) or (image_w.shape[1] < 100):
        return jsonify(code=4, response="Image1 too small"), 400
    if (image_id.shape[0] < 100) or (image_id.shape[1] < 100):
        return jsonify(code=4, response="Image2 too small"), 400
    code0, result0 = feature_extractor.extract_feature_array(image_id)

    code, result = feature_extractor.compare_faces(image_w,result0)

    if code == 0:
        if DEBUG:
            print(result)
        return jsonify(code=0, response="Success", identifications=result)
    else:
        return jsonify(code=code, response=result), 400
        

@app.route("/update-user-record", methods=['POST'])
def update_vector():
    """
    function to add/delete/update an in-memory vector and user record
    :return: json object with error code and response message
    """

    if request.method != "POST":
        return jsonify(code=1, response="Invalid request type"), 400

    data = request.get_json()
    try:
        uid = data["visitor_id"]
        request_type = data["request_type"]
        tenant_id = 1
        assert request_type in [0, 1, 2]  # 0 add 1 update 2 delete
    except (KeyError, AssertionError):
        return jsonify(code=2, response="Invalid data keys"), 400

    if request_type != 2:
        try:
            vector = data["vector"]
        except KeyError:
            return jsonify(code=2, response="Invalid data keys"), 400
    else:
        vector = None

    if tenant.update_database(uid, vector, request_type):
        return jsonify(code=0, response="Successfully updated in memory vector")
    else:
        if request_type == 0:
            return jsonify(code=4, response="User Id already present"), 400
        return jsonify(code=3, response="User Id not found"), 400


@app.route("/update-camera", methods=['POST'])
def update_camera():
    """
    function to call to add a new camera or refresh an existing camera
    :return: error code and response message
    """
    if request.method != "POST":
        return jsonify(code=1, response="Invalid request type"), 400

    data = request.get_json()
    print(data)
    try:
        camera_id = data["camera_id"]
        request_type = data["request_type"]
        tenant_id = 1
        assert request_type in [0, 1, 2]  # 0 add 1 update 2 delete
    except (KeyError, AssertionError):
        return jsonify(code=2, response="Invalid data keys"), 400

    if request_type != 2:
        try:
            url = data["camera_url"]
        except KeyError:
            return jsonify(code=2, response="Invalid data keys"), 400
    else:
        url = None

    if tenant.update_camera(camera_id, url, request_type):
        return jsonify(code=0, response="Successfully updated the camera")
    else:
        if request_type == 2:
            return jsonify(code=3, response="invalid camera id"), 400

        return jsonify(code=4, response="Unable to access camera"), 400


@app.route("/ocr_id", methods=['POST'])
def ocr_id():
    """
    API function used to extract the feature vector corresponding to a face passed in as base64 encoded string
    in the body of the request. Returns the vector if success. Returns error code + error message if failed
    :return: error code and the vector, in case of an error returns error code and error message
    """

    if request.method != "POST":
        return jsonify(code=1, response="Invalid request type"), 400

    data = request.get_json()

    try:
        img_str = data["img"]
    except KeyError:
        return jsonify(code=2, response="Invalid data keys"), 400

    try:
        img_str = bytes(img_str, 'utf-8')
        img_str = base64.b64decode(img_str)
        np_array = np.fromstring(img_str, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except (IOError, cv2.error, binascii.Error):
        return jsonify(code=3, response="Invalid image"), 400
    if (image.shape[0] < 100) or (image.shape[1] < 100):
        return jsonify(code=4, response="Image too small"), 400

    #save image into the space
    img_path = "id_img.png"
    cv2.imwrite( img_path , image )
    result = ocr.ocr( img_path , cls=True) 

    boxes = [line[0] for line in result] 
    txts = [line[1][0] for line in result] 
    scores = [line[1][1] for line in result] 

    out_dict ={}
    for i_num , ( i_box , i_txt , i_score) in enumerate(zip( boxes , txts , scores )):
        out_dict[ "text_{}".format( i_num ) ] = {
            "Text": i_txt ,
            "Bbox": str(i_box) ,
            "Score": str(i_score)
        }

    if( result is not None ):
        code =0 
    else:
        code = 400

    if code == 0:
        if DEBUG:
            print(result)
        return jsonify(code=200, response="Success", vector=out_dict)
    else:
        return jsonify(code=code, response=result), 400
    
if __name__ == '__main__':

    app.run(host='0.0.0.0', threaded=True, port=9000, debug=True)
