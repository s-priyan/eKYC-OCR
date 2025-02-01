import numpy as np
import torch, time
from torch.autograd import Variable
from .get_nets import PNet, RNet, ONet
from .box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from .first_stage import run_first_stage, run_first_stage_parallel, run_first_stage_parallel_half

DEBUG = False


def load_models():
    """
    Loads the face detection models and return
    :return: tuple of 3 networks
    """
    pnet = PNet().cuda()
    rnet = RNet().cuda()
    onet = ONet().cuda()
    pnet.eval()
    rnet.eval()
    onet.eval()
    return pnet, rnet, onet


def detect_faces(image, pnet, rnet, onet, min_face_size=20.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7]):
    """
    Arguments:
        Function to be called to detect faces in a single frame/ image. Uses the network passed in by ref.

        image: an instance of cv2 image.
        min_face_size: a float number that defines the size of the minimum face detectable.
        pnet: Proposal network
        rnet: Refinement network
        onet: output network
        thresholds: a list of length 3 corresponding to thresholds for each network.
        nms_thresholds: a list of length 3 corresponding to nms thresholds for each network's output.

    Returns:
        two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
        bounding boxes and facial landmarks.
    """

    # LOAD MODELS
    # pnet = PNet().cuda()
    # rnet = RNet().cuda()
    # onet = ONet().cuda()
    # onet.eval()

    # BUILD AN IMAGE PYRAMID
    # width, height = image.size
    height, width,_ = image.shape
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)
    # factor = 0.5

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size/min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m*factor**factor_count)
        min_length *= factor
        factor_count += 1
    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    t = time.time()
    if factor==0.5:
        bounding_boxes = run_first_stage_parallel_half(image, pnet, scale_list=scales, threshold=thresholds[0])
    elif factor==0.707:
        bounding_boxes = run_first_stage_parallel(image, pnet, scale_list=scales, threshold=thresholds[0])
    else:
        for s in scales:
            boxes = run_first_stage(image, pnet, scale=s, threshold=thresholds[0])
            if boxes is not None:
                bounding_boxes.append(boxes)
    if DEBUG:
        print("first stage", time.time() - t)
    t = time.time()
    # collect boxes (and offsets, and scores) from different scales
    # bounding_boxes = [i for i in bounding_boxes if i is not None]
    if len(bounding_boxes) == 0:
        return [], []
    bounding_boxes = np.vstack(bounding_boxes)

    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
    if DEBUG:
        print("nms", time.time() - t)
    t = time.time()
    bounding_boxes = bounding_boxes[keep]

    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    if DEBUG:
        print("calibrate", time.time() - t)
    t = time.time()
    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
    if DEBUG:
        print("convert to sq", time.time() - t)
    t = time.time()
    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, image, size=24)
    if DEBUG:
        print("get img boxes", time.time() - t)
    t = time.time()
    if len(img_boxes) == 0:
        return [], []
    with torch.no_grad():
        img_boxes = Variable(torch.FloatTensor(img_boxes).cuda())
        output = rnet(img_boxes)
    offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
    probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]
    if DEBUG:
        print("second stage", time.time() - t)
    t = time.time()
    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]

    keep = nms(bounding_boxes, nms_thresholds[1])
    if DEBUG:
        print("nms", time.time() - t)
    t = time.time()
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
    if DEBUG:
        print("calibrate+square", time.time() - t)
    t = time.time()
    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, image, size=48)
    if DEBUG:
        print("img boxes", time.time() - t)
    t = time.time()
    if len(img_boxes) == 0:
        return [], []
    with torch.no_grad():
        img_boxes = Variable(torch.FloatTensor(img_boxes).cuda())
        output = onet(img_boxes)
    landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
    offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
    probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]
    if DEBUG:
        print("stage 3", time.time() - t)
    t = time.time()
    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
    offsets = offsets[keep]
    landmarks = landmarks[keep]

    # compute landmark points
    width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
    height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
    xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
    landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
    landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
    bounding_boxes = bounding_boxes[keep]
    landmarks = landmarks[keep]
    if DEBUG:
        print("nms+op", time.time() - t)
    t = time.time()
    return bounding_boxes, landmarks

