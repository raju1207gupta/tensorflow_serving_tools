# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
from grpc.beta import implementations
import requests
import tensorflow as tf
import scipy
import numpy as np
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc,prediction_service_pb2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# The image URL is the location of the image we should send to the server
IMAGE_URL = 'https://tensorflow.org/images/blogs/serving/cat.jpg'

tf.compat.v1.app.flags.DEFINE_string('server', 'localhost:8500',
                                     'PredictionService host:port')
tf.compat.v1.app.flags.DEFINE_string('input_image', '',
                                     'path to image in JPEG format')
tf.compat.v1.app.flags.DEFINE_string('path_to_labels', '',
                                     'path to labels of classes')
FLAGS = tf.compat.v1.app.flags.FLAGS
def main():
    # Create stub
    host, port = FLAGS.server.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # Create prediction request object
    request = predict_pb2.PredictRequest()

    # Specify model name (must be the same as when the TensorFlow serving serving was started)
    request.model_spec.name = 'yolov4'

    # Initalize prediction 
    # Specify signature name (should be the same as specified when exporting model)
    request.model_spec.signature_name = ""
    request.inputs['inputs'].CopyFrom(
            tf.contrib.util.make_tensor_proto({FLAGS.input_image}))

    # Call the prediction server
    result = stub.Predict(request, 10.0)  # 10 secs timeout

    # Plot boxes on the input image
    category_index = label_map_util.load_labelmap(FLAGS.path_to_labels)
    boxes = result.outputs['detection_boxes'].float_val
    classes = result.outputs['detection_classes'].float_val
    scores = result.outputs['detection_scores'].float_val
    image_vis = vis_util.visualize_boxes_and_labels_on_image_array(
        FLAGS.input_image,
        np.reshape(boxes,[100,4]),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Save inference to disk
    scipy.misc.imsave('%s.jpg'%(FLAGS.input_image), image_vis)


if __name__ == '__main__':
  tf.compat.v1.app.run()
