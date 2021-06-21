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



server = 'localhost:8500'
host, port = server.split(':')
path_to_labels = "path_to/mscoco_label_map.pbtxt"

image_path = "path_to_image"
# height = image.shape[0]
# width = image.shape[1]
height = 720
width = 1280
image = cv2.imread(image_path)
image = cv2.resize(image,(width,height))
print("Image shape:", image.shape)

# create the RPC stub
channel = implementations.insecure_channel(host, int(port))
stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

# create the request object and set the name and signature_name params

request = predict_pb2.PredictRequest()
request.model_spec.name = 'faster_rcnn_resnet101'
#request.model_spec.version = '1'
request.model_spec.signature_name = 'detection_signature'

# fill in the request object with the necessary data
request.inputs['inputs'].CopyFrom(
tf.contrib.util.make_tensor_proto(image.astype(dtype=np.uint8), shape=[1, height, width, 3]))

# sync requests
result = stub.Predict(request, 30)

# Plot boxes on the input image
label_map = label_map_util.load_labelmap(path_to_labels)
categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=90,use_display_name=True)
category_index = label_map_util.create_category_index(categories)
boxes = result.outputs['detection_boxes'].float_val
classes = result.outputs['detection_classes'].float_val
scores = result.outputs['detection_scores'].float_val
try:
    if 'detection_masks' in result.outputs:
        mask_dict = result.outputs['detection_masks'].float_val
except:
    pass

boxes = np.reshape(boxes,[100,4])
boxes = np.squeeze(boxes)
scores = np.squeeze(scores)
classes = np.squeeze(classes)
indices = np.argwhere(classes == 1)
boxes = np.squeeze(boxes[indices])
scores = np.squeeze(scores[indices])
classes = np.squeeze(classes[indices])

image = vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    boxes,
    classes.astype(np.int32),
    scores,
    category_index,
    use_normalized_coordinates=True,
    line_thickness=3,
    min_score_thresh=0.5,
    skip_scores=True,
    skip_labels=True)

cv2.imshow("output",image)
cv2.imwrite("airport1_faster_rcnn_resnet101.jpg",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
    