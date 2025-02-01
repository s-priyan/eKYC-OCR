from paddleocr import PaddleOCR,draw_ocr 
import base64
import sys
import time
import binascii
import cv2
import numpy as np
from src.utils import  ocr_output , align_images
import base64
import io
import cv2
import matplotlib.pyplot as plt

def img_info_extractor( image ,  template  , roi_info , text_cleaner , engine=None   ):
    """
    API function used to extract the feature vector corresponding to a face passed in as base64 encoded string
    in the body of the request. Returns the vector if success. Returns error code + error message if failed
    :return: error code and the vector, in case of an error returns error code and error message
    """
    try:
        #image = cv2.imread( "../id_images/new_nic/rawimg8917.jpg" )
        print("[Info] aligning images...")
        aligned = align_images(image , template, debug=True )

        #save image into the space
        img_path = "id_comparison_img.png"
        cv2.imwrite( img_path , aligned )

        result = engine.ocr( img_path , det=True , rec=True , cls=True) 
        model_output = []
        
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

        result_out = ocr_output( roi_info , model_output  )

        # clean the ocr output
        if( text_cleaner ):
            result_clean = text_cleaner.text_sim_format( result_out )
        else:
            result_clean = result_out

        return result_clean
    
    except:
        return None