import json
import requests
import cv2
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

import matplotlib.image as mpimg

def preprocess(image):
    image = img_to_array(image, )
    data = cv2.resize(image, (416, 416))
    # im_arr = img_to_array(data, )
    im_arr = data/255.0
    im_arr = np.expand_dims(im_arr, axis=0)
    return im_arr

#img = mpimg.imread("D:/deeplearning learn/Automatic_License_plate_recognition/TensorFlow-2.x-YOLOv3/IMAGES/kite.jpg")
img = cv2.imread("D:/deeplearning learn/OrionEdgeSocialDistancingAPI/defect_folder/Wire-Loyalty52/MD/2021-02-08/4_4_Alert05-53-50.jpg")
#img = mpimg.imread("D:/deeplearning learn/marico_demo/airport_queue_management/queue4.jpg")
# create a json string to ask query to the depoyed model
img_expanded = preprocess(img)
#cv2.imdecode(np.fromstring(img, np.uint8), cv2.COLOR_RGB2BGR)
data = json.dumps({"signature_name": "serving_default",
                   "instances": img_expanded.tolist()})

# headers for the post request
headers = {"content-type": "application/json"}

# make the post request 
json_response = requests.post('http://localhost:8501/v1/models/yolov4-416/versions/1:predict',
                              data=data,
                              headers=headers)

# get the predictions
predictions = json.loads(json_response.text)
predictions = predictions['predictions']
for i, prediction in enumerate(predictions[0]):
    print("Prediction: ",np.argmax(prediction))
    