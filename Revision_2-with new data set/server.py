## SERVER_END_RUN

import cv2
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify, request
import flask 
import numpy as np 
import base64


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



src_model_snap = '/Users/dinithipurnadissnayake/Documents/Flask/train5000_val625_test_ALL_snapshot_rev2_new.h5'
 
 
app = flask.Flask(__name__) 
model = None
graph = None

def load_model(): 
    global model      
    model = tf.keras.models.load_model(src_model_snap)
    print("\n\nModel Loading Successful...")
    #global graph
    #graph = tf.compat.v1.get_default_graph()

def displaynm(num):
    arr=['dl_new_back', 'dl_new_front', 'nic_new_back', 'nic_new_front', 'nic_old_back', 'nic_old_front', 'pp_foreign', 'pp_local', 'unknown']
    return arr[num]

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
@app.route("/predict", methods = ["POST"]) 
def predict(): 
    data = {} # dictionary to store result 
    data['code'] = 1
    # Check if image was properly sent to our endpoint 
    if request.method == "POST":
        if request.json['image']: 
            image = request.json['image']
            #image = flask.request.files["image"].read()
            image_tensor = prepare_image(image, b64encoded=True)
            if image_tensor is None:
                print("Image Loading Failed..")
                data['response'] = 'Image Reading Error: Image Loading Failed'
            else:
                #with graph.as_default():
                #    prediction = model.predict(image_tensor).astype(np.float32)
                #load_model()
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
        else:
            data['response'] = 'Image not entered: Image Loading Failed'
    else:
        data['response'] = 'Request Type ERROR'

    return jsonify(data)

  
if __name__ == "__main__": 
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started")) 
    load_model() 
    app.run()