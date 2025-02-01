import boto3
import botocore
import cv2
BUCKET_NAME = 'adl-vision-rnd-test' # replace with your bucket name
#KEY = 'my_image_in_s3.jpg' # replace with your object key
s3 = boto3.resource('s3')
s3.Bucket(BUCKET_NAME).download_file("sahan/20210928_133129.jpg", "/home/ubuntu/home/eKYC_s3/Data/Face_compare_IMG/my_local_image1.jpg")