# CLIENT_END_RUN
#python client.py /home/adldata/NIC_PROJECT/NEW_DATA/inf_images/dog.jpg
import requests 
import sys
import base64
  
URL = "http://localhost:5000/predict"
  
# provide image name as command line argument 
IMAGE_PATH = sys.argv[1]  

image = base64.b64encode(open(IMAGE_PATH, "rb").read()) 
payload = {"image": image} 

# make request to the API 
request = requests.post(URL, files = payload).json() 
  
if not request["code"]: 
    # Print formatted Result 
    print("Client end Prediction : ", request['prediction'],"---Check HARSHA")
else:
    print("Request failed")