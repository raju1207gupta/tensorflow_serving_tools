from __future__ import print_function
from PIL import Image
from grpc.beta import implementations
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import requests
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import cv2
from keras.preprocessing.image import load_img, img_to_array
from yolo_post_processing import *

class_names = ['person',
                'bicycle',
                'car',
                'motorbike',
                'aeroplane',
                'bus',
                'train',
                'truck',
                'boat',
                'traffic light',
                'fire hydrant',
                'stop sign',
                'parking meter',
                'bench',
                'bird',
                'cat',
                'dog',
                'horse',
                'sheep',
                'cow',
                'elephant',
                'bear',
                'zebra',
                'giraffe',
                'backpack',
                'umbrella',
                'handbag',
                'tie',
                'suitcase',
                'frisbee',
                'skis',
                'snowboard',
                'sports ball',
                'kite',
                'baseball bat',
                'baseball glove',
                'skateboard',
                'surfboard',
                'tennis racket',
                'bottle',
                'wine glass',
                'cup',
                'fork',
                'knife',
                'spoon',
                'bowl',
                'banana',
                'apple',
                'sandwich',
                'orange',
                'broccoli',
                'carrot',
                'hot dog',
                'pizza',
                'donut',
                'cake',
                'chair',
                'sofa',
                'pottedplant',
                'bed',
                'diningtable',
                'toilet',
                'tvmonitor',
                'laptop',
                'mouse',
                'remote',
                'keyboard',
                'cell phone',
                'microwave',
                'oven',
                'toaster',
                'sink',
                'refrigerator',
                'book',
                'clock',
                'vase',
                'scissors',
                'teddy bear',
                'hair drier',
                'toothbrush']
def preprocess(image):
    image = img_to_array(image, )
    data = cv2.resize(image, (416, 416))
    # im_arr = img_to_array(data, )
    im_arr = data/255.0
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr


def post_process(result,im_hight=720,im_width=1280, threshold =0.6):
    predictions = result.outputs['output']
    predictions = tf.make_ndarray(predictions)
    img_arr = np.squeeze(predictions, 0)
    path = "D:/tf-serving/Yolo_deployment/yolov2-tiny.cfg"
    meta = parse_cfg(path=path)
    meta.update({"labels": class_names})

    boxes = box_contructor(meta=meta, out=img_arr, threshold=threshold)

    boxesInfo = list()
    score_list = []
    classes_index = []
    print(len(boxes))
    for box in boxes:
        tmpBox, prob_,final_box,class_indx = process_box(b=box, h=im_hight, w=im_width, threshold=threshold, meta=meta)

        if tmpBox is None:
            continue
        boxesInfo.append(final_box)
        score_list.append(prob_)
        classes_index.append(class_indx)
    return boxesInfo,score_list,classes_index

server = 'localhost:8500'
host, port = server.split(':')
path_to_labels = "path_to/mscoco_label_map.pbtxt"
path_to_video = "path"
#image = np.array(Image.open("D:/deeplearning learn/Automatic_License_plate_recognition/TensorFlow-2.x-YOLOv3/IMAGES/street.jpg"))
# height = image.shape[0]
# width = image.shape[1]
height = 720
width = 1280
# print("Image shape:", image.shape)

# create the RPC stub
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)


video = cv2.VideoCapture(path_to_video)
frame_no = 0
fps=int(video.get(cv2.CAP_PROP_FPS))
# create the request object and set the name and signature_name params
while True:
    ret, image = video.read()
    frame_no=frame_no+1
    image = cv2.resize(image,(width,height))
    (h, w) = image.shape[:2]
    if ret and frame_no>fps:
        frame_no = 0
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'yolov2'
        request.model_spec.signature_name = 'predict'
        # fill in the request object with the necessary data
        request.inputs['input'].CopyFrom(
        tf.contrib.util.make_tensor_proto(preprocess(image), dtype=tf.float32))

        # sync requests
        result = stub.Predict(request, 3)
        out_boxes,out_scores,out_classes = post_process(result,im_hight=h,im_width=w,threshold=0.3)
        for i, c in reversed(list(enumerate(out_classes))):
            box = out_boxes[i]
            score = round(out_scores[i],2)
            if score > 0.3 :
                top, left, bottom, right = box
                # startY = int(max(0, np.floor(top + 0.5).astype('int32')))
                # startX = int(max(0, np.floor(left + 0.5).astype('int32')))
                # endY = int(min(h, np.floor(bottom + 0.5).astype('int32')))
                # endX = int(min(w, np.floor(right + 0.5).astype('int32')))
                (startX, startY) = (max(0, int(left)), max(0, int(top)))
                (endX, endY) = (min(w - 1, int(right)), min(h - 1, int(bottom))) 
                
                if class_names[c] == 'person':
                    color = (255,0,0)
                    cv2.putText(image, class_names[c], (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        cv2.imshow("output",image)
    if cv2.waitKey(1) == ord('q'):
        video.release()
        cv2.destroyAllWindows()
        break
    

video.release()
cv2.destroyAllWindows()