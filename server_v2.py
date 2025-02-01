## SERVER_END_RUN

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

log_format = "%(asctime)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(log_format)

logger = logging.getLogger('my_app')
logger.setLevel(logging.INFO)

logHandler = handlers.TimedRotatingFileHandler('biometric.log', when='midnight', interval=1)
logHandler.setLevel(logging.INFO)
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)


# logname = "biometric.log"
# handler = TimedRotatingFileHandler(logname, when="midnight", interval=1)
# handler.suffix = "%Y%m%d"

# logger.addHandler(handler)

# today = date.today()
# logging.basicConfig(filename='log.log', encoding='utf-8', level=logging.DEBUG)
# logging.debug('This message should go to the log file')
# logging.info('So should this')
# logging.warning('And this, too')
# logging.error('And non-ASCII stuff, too, like Øresund and Malmö')


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



src_model_snap = 'train5000_val625_test_ALL_snapshot.h5'
 
 
app = flask.Flask(__name__) 
model = None
graph = None

def load_model(): 
    global model      
    model = tf.keras.models.load_model(src_model_snap)
    print("\n\nModel Loading Successful...")
    #global graph
    #graph = tf.compat.v1.get_default_graph()


def prepare_image(img_bstr, target_size=(224,224), b64encoded = False):
    if b64encoded:
        img_bstr = base64.b64decode(img_bstr)
    file_bytes = np.asarray(bytearray(img_bstr), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        print("\n\nImage loading FAILED\n\n")
        return None
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
        image_tensor = tf.image.resize([img], target_size)
        return image_tensor

# Now, we can predict the results. 
@app.route("/predict",methods = ["POST"]) 
def predict(): 
    data = {} # dictionary to store result 
    data['code'] = 1
    # Check if image was properly sent to our endpoint 
    if request.method == "POST":
        if request.json['image']: 
            image = request.json['image']
            req_id= request.json['id']
            #image = flask.request.files["image"].read()
            image_tensor = prepare_image(image, b64encoded=True)
            if image_tensor is None:
                print("Image Loading Failed..")
                data['response'] = 'Image Reading Error: Image Loading Failed'
                
                log_message=req_id+"    "+data['response']
                logger.info(log_message)
                return jsonify(data) , 400
            else:
                #with graph.as_default():
                #    prediction = model.predict(image_tensor).astype(np.float32)
                prediction = model.predict(image_tensor).astype(np.float32)
                data['code'] = 0
                data['response'] = 'Model prediction PASSED'
                print("\n\nServer end result: ", prediction, "---Check HARSHA")
                data['prediction'] = prediction.tolist()
                temp1='    '
                for r in data['prediction']:
                    temp1+= str(r)+' '
                log_message=req_id+"    "+data['response'] +temp1
                logger.info(log_message)
                return jsonify(data),200
        else:
            req_id= request.json['id']
            data['response'] = 'Image not entered: Image Loading Failed'
            log_message=req_id+"    "+data['response']
            logger.info(log_message)
            return jsonify(data) ,400
    else:
        data['response'] = 'Request Type ERROR'
        log_message="    "+data['response']
        logger.info(log_message)
        return jsonify(data) , 405
    
    

  
if __name__ == "__main__": 
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started")) 
    load_model() 
    app.run(host='0.0.0.0', threaded=True, port=80)
