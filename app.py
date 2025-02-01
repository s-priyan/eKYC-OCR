# Import for id validation
import cv2
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request
import flask 
import numpy as np 
import base64
import logging
import logging
from datetime import date
from logging.handlers import TimedRotatingFileHandler
import logging
import logging.handlers as handlers
import time
#### Encryption
from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, create_refresh_token, get_jwt_identity
from datetime import timedelta

# Import for paddle ocr
from paddleocr import PaddleOCR,draw_ocr 
import base64
import sys
import time
import binascii
import cv2
import numpy as np
from flask import Flask, jsonify, request
from src.utils import select_template_sim_roi , select_template_roi , align_images , ocr_output
from src.roi_extract import roi_dl , roi_new_nic , roi_passport , roi_new_nic_back
from src.text_processing import New_Nic_Text
from src.id_info_extractor import img_info_extractor
import base64
import io
import cv2
from imageio import imread
import matplotlib.pyplot as plt
from src.text_processing import New_Nic_Text , Driving_Text
# Paddleocr supports Chinese, English, French, German, Korean and Japanese. 

# You can set the parameter `lang` as `ch`, `en`, `french`, `german`, `korean`, `japan` 

# to switch the language model in order. 
import base64
import sys
import time
import binascii
import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request,send_file
from paddleocr import PaddleOCR

from src.config_parser import conf
from src.feature_extractor import FeatureExtractor
from src.my_logging import Logger
from src.video_capture import VideoCaptureAsync
from src.visitortags_api_access_functions import api_connector
from src.tenent import Tenant

##for s3 integration
import boto3
import botocore
import os
BUCKET_NAME = 'adl-vision-rnd-test' # replace with your bucket name
#KEY = 'my_image_in_s3.jpg' # replace with your object key
s3 = boto3.resource('s3')


import time

# Logging for id validation
log_format = "%(asctime)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)

logger = logging.getLogger('my_app')
logger.setLevel(logging.INFO)



logHandler = handlers.TimedRotatingFileHandler('et_telco-log/biometric.log', when='midnight', interval=1)
logHandler.setLevel(logging.INFO)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
token_init_time=time.time()
# Id validation model gpu configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

app = Flask(__name__)
DEBUG = True
access_token_timeout_mins=5
refresh_token_timeout_mins=5
app.config["JWT_SECRET_KEY"] = "super-secret"
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(minutes=access_token_timeout_mins*60)
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = timedelta(days=refresh_token_timeout_mins)
biometric_user = "biomet"
biometric_pw = "bio@123"
jwt = JWTManager(app)

fr_path = conf.hyper_params.fr_model.path

visitor_db = api_connector.load_visitor_db()
camera_db = api_connector.load_cam_db()

tenant = Tenant(1, camera_db, visitor_db)
tenant.initiate()

feature_extractor = FeatureExtractor(fr_path)

"""
initialize the paddleocr engine with the appropriae detction , recognition , classification models
"""
ocr =PaddleOCR(use_angle_cls=True,use_gpu=True ,lang="en" ,
                det_model_dir='/home/ubuntu/home/eKYC_s3/ch_ppocr_server_v2.0_det_infer' , 
                rec_model_dir='/home/ubuntu/home/eKYC_s3/en_number_mobile_v2.0_rec_infer')  # need to run only once to download and load model into memory 

print("[Info] loading template image...")

target_dict={
    'dl_new_back':0 ,
    'dl_new_front':1 ,
    'nic_new_back':2,
    'nic_new_front':3 ,
    'nic_old_back':4 ,
    'nic_old_front':5 ,
    'pp_foreign':6 ,
    'pp_local':7 ,
    'unknown':8 }

def displaynm(num):
    arr=['dl_new_back', 'dl_new_front', 'nic_new_back', 'nic_new_front', 'nic_old_back', 'nic_old_front', 'pp_foreign', 'pp_local', 'unknown']
    return arr[num]


src_model_snap = 'train5000_val625_test_ALL_snapshot.h5'
model = None
graph = None

def load_model(): 
    global model      
    model = tf.keras.models.load_model(src_model_snap)
    print("\n\nModel Loading Successful...")
    #global graph
    #graph = tf.compat.v1.get_default_graph()


def prepare_image( img , target_size=(224,224), b64encoded = False):
    try:
        if img is None:
            print("\n\nImage loading FAILED\n\n")
            return None
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
            image_tensor = tf.image.resize([img], target_size)
            return image_tensor
    except:
        return None
        
def get_expire(T,duration):
    months=[1,3,5,7,8,10,12]
    T[4]+=duration
    if(T[4]>59):
        T[3]+=1
        T[4]-=60
        if T[3]>24:
            T[3]-=24
            T[2]+=1
            if(T[2]>30 and T[2] not in months):
                T[1]+=1
                T[2]-=30
            elif(T[2]>31 and T[2] in months):
                T[1]+=1
                T[2]-=31
            if(T[1]>12):
                T[0]+=1
                T[1]-=12
    return time.asctime(time.struct_time(T))
    
def get_expire_days(T,duration):
    months=[1,3,5,7,8,10,12]
    T[2]+=duration
    if(T[2]>30 and T[2] not in months):
        T[1]+=1
        T[2]-=30
    elif(T[2]>31 and T[2] in months):
        T[1]+=1
        T[2]-=31
    if(T[1]>12):
        T[0]+=1
        T[1]-=12
    return time.asctime(time.struct_time(T))
            
        

# Now, we can predict the results.
@app.route("/token", methods=["POST"])
def create_token():
    data_raw=request.get_json()
    data=data_raw["requestHeader"]
    username = str(data["consumerKey"])
    password = str(data["consumerSecret"])
    #password = request.json.get("password", None)
    if (username == biometric_user) and (password == biometric_pw):
        access_token = create_access_token(identity=username, fresh=True)
        refresh_token = create_refresh_token(identity=username)
        expire=get_expire(list(time.localtime(time.time())),access_token_timeout_mins)
        token_init_time=time.time()
        response={"tokenDetails": {"deviceId": data["deviceId"],"accessToken":access_token ,"expiresIn":expire,"refreshToken": refresh_token},"responseHeader": {"requestId": data["requestId"],"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "200","responseDesc": "Success"}}
        return jsonify(response),200
    response={"responseHeader": {"requestId": data["requestId"],"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "400","responseDesc": "Bad username or password"}}
    return jsonify(response), 400

    
    
@app.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    data_raw=request.get_json()
    data=data_raw["requestHeader"]
    username = str(data["consumerKey"])
    password = str(data["consumerSecret"])
    if (username == biometric_user) and (password == biometric_pw):
        identity = get_jwt_identity()
        access_token = create_access_token(identity=identity, fresh=True)
        response={"tokenDetails": {"deviceId": data["deviceId"],"accessToken":access_token ,"expiresIn":expire,"refreshToken": refresh_token},"responseHeader": {"requestId": data["requestId"],"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "200","responseDesc": "Success"}}
        return jsonify(response),200
    response={"responseHeader": {"requestId": data["requestId"],"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "400","responseDesc": "Bad username or password"}}
    return jsonify(response), 400
    
    

@app.route("/upload", methods=["POST"])
@jwt_required(fresh=True)
def upload():
    if request.method != "POST":
        return jsonify({"responseHeader": {"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "405","responseDesc":"Method not allowed"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 405
    try:
        data= request.files['img']
        
        if(data.filename == ''):
            return jsonify({"responseHeader": {"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "400","responseDesc":"Bad Request"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}),400
            
            
        pre=str(request.form['docType'])
        request_id=request.form['requestId']
        filename=str(time.time())
    
        filename=pre+filename.replace('.','')
        data.save("Data/temp/"+filename+'.jpg')
        response={"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "200","responseDesc":"Success"},"file_reference":filename,"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
        return jsonify(response), 200
    except KeyError:
        response={"responseHeader": {"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "400","responseDesc":"Bad Request Format"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
        return jsonify(response),400


@app.route('/get-image')
def get_image():
    
    #if request.method != "POST":
        #return jsonify({"responseHeader": {"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "405","responseDesc":"Method not allowed"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 405
    data = request.get_json()
    try:
        request_head=data["requestHeader"]
        request_id=request_head["requestId"]
        img_id = data["Reference"]
    except KeyError:
        response={"responseHeader": {"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "400","responseDesc":"Bad Request Format"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
    img=cv2.imread("Data/temp/"+str(img_id)+".jpg")
          
    return send_file("Data/temp/"+str(img_id)+'.jpg', mimetype="Data/temp")
    
@app.route('/get-image_web')
def get_image_web():
    reference = str(request.args.get('reference'))
    request_id = str(request.args.get('requestId'))
    #if request.method != "GET":
     #   return jsonify({"responseHeader": {"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "405","responseDesc":"Method not allowed"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 405
          
    return send_file("Data/temp/"+str(reference)+'.jpg', mimetype="image/jpg")


@app.route("/delete", methods=["POST"])
@jwt_required(refresh=True)
def delete():
    if request.method != "POST":
        return jsonify({"responseHeader": {"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "405","responseDesc":"Method not allowed"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    try:
        data= request.form['file_reference']
        
        if(data== ''):
            return jsonify({"responseHeader": {"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "400","responseDesc":"Bad Request"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}),400
            
            
        
        request_id=request.form['requestId']
        os.remove("Data/temp/"+str(data)+'.jpg')
        response={"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "200","responseDesc":"Success"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
        return jsonify(response), 200
    except KeyError:
        response={"responseHeader": {"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": "400","responseDesc":"Bad Request Format"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
        return jsonify(response),400


@app.route("/compare-ID",methods = ["POST"]) 
@jwt_required(fresh=True)
def compare_ID():
    if request.method != "POST":
        return jsonify(code=1, response="Invalid request type", token_timeout=access_token_timeout_mins*60-(time.time()-token_init_time)), 400

    data = request.get_json()

    try:
        request_head=data["requestHeader"]
        request_id=request_head["requestId"]
        img_id1_ref = data["ReferenceIDBase"]
        img_id2_ref = data["ReferenceIDUser"]

    except KeyError:
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "2","responseDesc": "Invalid data keys"},
                        "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    try:
        
        image_id1 = cv2.imread("Data/temp/"+str(img_id1_ref)+".jpg")
        image_id2 = cv2.imread("Data/temp/"+str(img_id2_ref)+".jpg")

        # get id types
        
        image_tensor_id1 , image_tensor_id2 = prepare_image(image_id1 , b64encoded=True) , prepare_image(image_id2 , b64encoded=True)
        
        if (image_tensor_id1 is None) or (image_tensor_id2 is None):
            
           print("Image Loading Failed..")
           data['response'] = 'Image Reading Error: Image Loading Failed'             
           log_message=req_id+"    "+data['response']
           logger.info(log_message)

           return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "1","responseDesc": "Image Reading Error: Image Loading Failed"},
                            "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    
        else:
           
           #with graph.as_default():
           #prediction = model.predict(image_tensor).astype(np.float32)

           tensor_batch = tf.concat( (image_tensor_id1 , image_tensor_id2) , axis=0 ) 
           predictions = model.predict(tensor_batch).astype(np.float32)          
           y_pred_id1 , y_pred_id2 = np.argmax(predictions[0], axis=0) , np.argmax(predictions[1], axis=0)
           type_id1 , type_id2 = displaynm(y_pred_id1) , displaynm(y_pred_id2)
           id_1_type , id_2_type = target_dict[type_id1] , target_dict[type_id2]

    except (IOError, cv2.error, binascii.Error):
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "3","responseDesc": "Invalid image"},
                            "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    
    if (image_id1.shape[0] < 100) or (image_id1.shape[1] < 100):
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "4","responseDesc": "Image1 too small"},
                            "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    
    if (image_id2.shape[0] < 100) or (image_id2.shape[1] < 100):
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "4","responseDesc": "Image2 too small"},
                            "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    
    if( id_1_type  != id_2_type ):
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "5","responseDesc": "Different ID formats"},
                            "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    
    code0, result0 = feature_extractor.extract_feature_array(image_id2)
    print(result0)
    if( (code0 != 5) and (code0 != 6) ): 
        code, result = feature_extractor.compare_faces(image_id1,result0)
    else:
        code = 6

    # OCR based id number comparison
    print("[Info] loading template and roi info ")

    template  , roi_info , text_cleaner = select_template_sim_roi( id_1_type )

    # id information extraction
    result_img_1 = img_info_extractor(  image_id1 , template , roi_info , text_cleaner , ocr )
    result_img_2 = img_info_extractor(  image_id2 , template , roi_info , text_cleaner , ocr )
    print( result_img_1 , result_img_2 )
    similarity_score = 0

    if( ( result_img_1 is None ) or ( result_img_2 is None ) ):
        similarity_score = -1
    else:
        for i_key in result_img_2.keys() :
            if( result_img_1[i_key] == result_img_2[i_key]  ):
                similarity_score += 1

    if code == 0:

        response={"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": str(code),"responseDesc": "SUCCESS"},
                    "similarityScore": result["0_conf"],"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
        return jsonify(response)
    
    else:
        if (similarity_score >0):
 
            response={"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": str(0),"responseDesc": "SUCCESS"},
                    "similarityScore": 1.0,"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
            return jsonify(response)

        elif( similarity_score !=-1 )  :

            response={"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": str(0),"responseDesc": "SUCCESS"},
                    "similarityScore": 0.0 ,"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
            return jsonify(response)
        
        else:

            result={"similarity measure failed"}
            return jsonify({"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": str(code),
                "responseDesc": "SUCCESS"},"Response": result,"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400


 
@app.route("/predict",methods = ["POST"]) 
@jwt_required(fresh=True)
def predict(): 
    data = {} # dictionary to store result 
    data['code'] = 1
    # Check if image was properly sent to our endpoint 
    data_n = request.get_json()
    if request.method == "POST":
        #if request.json['image']: 
        try:
            request_head=data_n["requestHeader"]
            request_id=request_head["requestId"]
            img_id = data_n["IDImageReference"]
            image=cv2.imread("Data/temp/"+str(img_id)+".jpg")
            image_tensor = prepare_image(image, b64encoded=True)
        except KeyError:
            return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "2","responseDesc": "Invalid data keys"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
        if image_tensor is None:
            print("Image Loading Failed..")
            data['response'] = 'Image Reading Error: Image Loading Failed'
            return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "3","responseDesc": "Image Reading Error: Image Loading Failed"},"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
        
        else:
        #with graph.as_default():
        #    prediction = model.predict(image_tensor).astype(np.float32)
            prediction = model.predict(image_tensor).astype(np.float32)
                
            data['code'] = 0
            data['response'] = 'Model prediction PASSED'
            print("\n\nServer end result: ", prediction, "---Check HARSHA")
                
            y_pred = np.argmax(prediction, axis=1)
            types=displaynm(y_pred[0])
            val=prediction.tolist()
            predictionlist={}
            predictionlist["class"]= types
            predictionlist["probability"]=val[0][y_pred[0]]
            data['prediction'] = predictionlist
            response={"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": str(0),"responseDesc": "SUCCESS"},"Prediction": predictionlist,"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
            return jsonify(response),200
       
    else:
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "1","responseDesc": "Method Not Allowed"},
                        "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 405

        

@app.route("/compare-image", methods=['POST'])
@jwt_required(fresh=True)
def compare_image():
    """
    API function to run comparison in two images (web cam captured and id). The images should be passed in as
    base64 encoded string in the body of the request (JSON format). The image is compared against the each other
    and confidence is returned.
    :return: confidence level
    """
    #response={"responseHeader": {"requestId": "e54eb678-b7b2-11ea-b3de-0242ac130004","timestamp": "2021-09-10T10:36:58.172","responseCode": "200","responseDesc": "SUCCESS"},"similarityScore": 88.2}
    if request.method != "POST":
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "1","responseDesc": "Method Not Allowed"},
                        "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 405

    data = request.get_json()
    

    try:
        request_head=data["requestHeader"]
        request_id=request_head["requestId"]
        img_webc = data["clientImageReference"]
        img_id = data["baseImageReference"]

    except KeyError:
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "2","responseDesc": "Invalid data keys"},
                        "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    try:
        
        image_w = cv2.imread("Data/temp/"+str(img_webc)+".jpg")
        image_id = cv2.imread("Data/temp/"+str(img_id)+".jpg")
          

    except (IOError, cv2.error, binascii.Error):
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "3","responseDesc": "Invalid Image"},
                        "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    if (image_w.shape[0] < 100) or (image_w.shape[1] < 100):
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "4","responseDesc": "Image1 too small"},
                        "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
                        
    if (image_id.shape[0] < 100) or (image_id.shape[1] < 100):
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "5","responseDesc": "Image2 too small"},
                    "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    code0, result0 = feature_extractor.extract_feature_array(image_id)
 
    code, result = feature_extractor.compare_faces(image_w,result0)

    if code == 0:
        if DEBUG:
            print(result)
        response={"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": str(code),"responseDesc": "SUCCESS"},"similarityScore": result["0_conf"],"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
        return jsonify(response)
    else:
        return jsonify({"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": str(code),
                "responseDesc": "SUCCESS"},"Response": result,"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400

@app.route("/ocr-id", methods=['POST'])
@jwt_required(fresh=True)
def ocr_img():
    """
    API function used to extract the feature vector corresponding to a face passed in as base64 encoded string
    in the body of the request. Returns the vector if success. Returns error code + error message if failed
    :return: error code and the vector, in case of an error returns error code and error message
    """
    data = {} # dictionary to store result 
    data['code'] = 1
    if request.method != "POST":
        return jsonify(code=1, response="Invalid request type",token_timeout=access_token_timeout_mins*60-(time.time()-token_init_time)), 405

    data = request.get_json()
    if DEBUG:
        print(data)

    try:
        request_head=data["requestHeader"]
        doct_type=data["docType"]
        img_id =  data["referenceNo"]
        request_id =  request_head["requestId"]

    except KeyError:
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "2","responseDesc": "Invalid data keys"},
                        "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400

    try:

        image_id = cv2.imread("Data/temp/"+str(img_id)+".jpg")
        #image = flask.request.files["image"].read()
        image_tensor = prepare_image( image_id , b64encoded=True)
        if image_tensor is None:
            
           print("Image Loading Failed..")
           data['response'] = 'Image Reading Error: Image Loading Failed'             
           log_message=req_id+"    "+data['response']
           logger.info(log_message)
           return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "1","responseDesc": "Image Reading Error: Image Loading Failed"},
                            "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
        else:
           
           #with graph.as_default():
           #    prediction = model.predict(image_tensor).astype(np.float32)
           prediction = model.predict(image_tensor).astype(np.float32)              
           data['code'] = 0
           data['response'] = 'Model prediction PASSED'
           print("\n\nServer end result: ", prediction, "---Check HARSHA")
                
           y_pred = np.argmax(prediction, axis=1)
           types=displaynm(y_pred[0])
           id_type = target_dict[types]

    except (IOError, cv2.error, binascii.Error):
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "3","responseDesc": "Invalid image"},
                            "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400


    if (image_id.shape[0] < 100) or (image_id.shape[1] < 100):
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "4","responseDesc": "Image1 too small"},
                            "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400

    print("[Info] loading template and roi info ")

    template  , roi_info , text_cleaner = select_template_roi( id_type )
    #image = cv2.imread( "../id_images/new_nic/rawimg8917.jpg" )
    print("[Info] aligning images...")
    aligned = align_images(image_id, template, debug=True )
    model_output = []
    try:
        #save image into the space
        img_path = "id_img.png"
        cv2.imwrite( img_path , aligned )

        result = ocr.ocr( img_path , det=True , rec=True , cls=True) 

        for i_pred in result :
            i_bbox = i_pred[0]
            i_txt , i_conf = i_pred[1][0] , i_pred[1][1]

            x1 , y1 , x2 , y2 = int(i_bbox[0][0]) , int(i_bbox[0][1]) , int(i_bbox[1][0]) , int(i_bbox[1][1])
            x3 , y3 , x4 , y4 = int(i_bbox[2][0]) , int(i_bbox[2][1]) , int(i_bbox[3][0]) , int(i_bbox[3][1])

            i_sample={
                "text": i_txt ,
                "i_conf":i_conf , 
                "bbox":[ x1 , y1 , x3 , y3  ] 
            }
            model_output.append( i_sample )

        t2 = time.time()

    except:
        return jsonify({"responseHeader": {"timestamp":time.asctime( time.localtime(time.time()) ) ,"responseCode": "4","responseDesc": "OCR Error ! , Input Clear Image"},
                            "token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}), 400
    
    result_out = ocr_output( roi_info , model_output  )
    
    # clean the ocr output
    if( text_cleaner ):
        result_clean = text_cleaner.text_format( result_out )
    else:
        result_clean = result_out

    if( result_out is not None ):
        code =0 
    else:
        code = 400

    if code == 0:
        if DEBUG:
            print(result_clean)
        result_clean["detectedDocument"], result_clean["city"] , result_clean["district"]= types , 'null' , 'null'
        response={"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": str(code),"responseDesc": "SUCCESS"},
                    "documentDetectionResponse": result_clean ,"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
        return jsonify(response)

    else:
        response={"responseHeader": {"requestId": request_id,"timestamp": time.asctime( time.localtime(time.time()) ),"responseCode": str(code),"responseDesc": "SUCCESS"},
                    "documentDetectionResponse": "Input Clear Image" ,"token_timeout":access_token_timeout_mins*60-(time.time()-token_init_time)}
        return jsonify(response)


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
                "please wait until server has fully started")) 
    load_model() 
    app.run(host='0.0.0.0', threaded=True, port=9000 , debug=True)
