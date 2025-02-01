import numpy as np
import cv2
from src.roi_extract import roi_dl , roi_new_nic , roi_passport , roi_new_nic_back , roi_sim_dl , roi_sim_new_nic , roi_sim_passport , roi_sim_old_nic
from src.text_processing import New_Nic_Text , Driving_Text

def enhance_image( image ):
    try:
        # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        thresh = cv2.adaptiveThreshold(blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15 , 25)

        img_out = image + cv2.bitwise_not( thresh ).reshape( image.shape[0] ,image.shape[1],1)

        return img_out
    
    except :
        print("Image Enhancement Error !!")
        return None

def align_images(image, template, maxFeatures=4500, keepPercent=0.2, debug=False):

    try:
        imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(maxFeatures)
        (kpsA, descsA) = orb.detectAndCompute(imgGray, None)
        (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
        
        method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
        matcher = cv2.DescriptorMatcher_create(method)
        matches = matcher.match(descsA, descsB, None)

        matches = sorted(matches, key=lambda x: x.distance)

        keep = int(len(matches) * keepPercent)
        matches = matches[:keep]
        
        ptsA = np.zeros((len(matches), 2), dtype=float)
        ptsB = np.zeros((len(matches), 2), dtype=float)

        for (i, m) in enumerate(matches):
            ptsA[i] = kpsA[m.queryIdx].pt
            ptsB[i] = kpsB[m.trainIdx].pt
        
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

        (h, w) = template.shape[:2]
        aligned = cv2.warpPerspective(image, H, (w, h))
        aligned = enhance_image( aligned )
        cv2.imwrite( "aligned_image.jpg" ,  aligned )

        return aligned

    except :
        print("Alignement Error !!!")
        return None

def bb_intersection_over_union(boxA, boxB):

    try :
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    except :
        print("IOU computation Error !!!")
        return None

def ocr_output( samples , model_output ):
    try:
        target_results = {}
        iou_thresh = 0.05
        for i_want in samples :
            ious = []

            for i_pred in model_output :
                ious.append(  bb_intersection_over_union( i_want['bbox']  , i_pred['bbox']  ) )

            i_index = np.argmax( ious )
            iou_value = ious[ i_index ]
            if iou_value > iou_thresh :
                target_results[i_want['id']] = model_output[i_index]['text']  

            else:
                target_results[i_want['id']] = "None" 

        return target_results
    except :
        return None


def select_template_sim_roi( id_type ):
    if( id_type==1 ):
        input_template = "/home/ubuntu/home/eKYC_s3/templates_similarity/DL_id1.png"
        template = cv2.imread( input_template )
        roi_info = roi_sim_dl
        text_cleaner = Driving_Text()

    elif( id_type==7 ):
        input_template = "/home/ubuntu/home/eKYC_s3/templates_similarity/Passport_id.jpg"
        template = cv2.imread( input_template )
        roi_info = roi_sim_passport
        text_cleaner = None

    elif( id_type==3 ):
        input_template = "/home/ubuntu/home/eKYC_s3/templates_similarity/New_Nic_id.png"
        template = cv2.imread( input_template )
        roi_info = roi_sim_new_nic
        text_cleaner = New_Nic_Text()
    
    elif( id_type==5 ):
        input_template = "/home/ubuntu/home/eKYC_s3/templates_similarity/Old_Nic_id.jpeg"
        template = cv2.imread( input_template )
        roi_info = roi_sim_old_nic
        text_cleaner = None

    return template , roi_info , text_cleaner

def select_template_roi( id_type ):
    
    if( id_type==1 ):
        input_template = "/home/ubuntu/home/eKYC_s3/templates/DL_id1.png"
        template = cv2.imread( input_template )
        roi_info = roi_dl
        text_cleaner = Driving_Text()

    elif( id_type==7 ):
        input_template = "/home/ubuntu/home/eKYC_s3/templates/Passport_id.jpg"
        template = cv2.imread( input_template )
        roi_info = roi_passport
        text_cleaner = None

    elif( id_type==3 ):
        input_template = "/home/ubuntu/home/eKYC_s3/templates/New_Nic_id.png"
        template = cv2.imread( input_template )
        roi_info = roi_new_nic
        text_cleaner = New_Nic_Text()

    elif( id_type==2 ):
        input_template = "/home/ubuntu/home/eKYC_s3/templates/New_Nic_id_Back.jpeg"
        template = cv2.imread( input_template )
        roi_info = roi_new_nic_back
        text_cleaner = None

    return template , roi_info , text_cleaner