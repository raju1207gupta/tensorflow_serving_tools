import tensorflow as tf

# Assuming object detection API is available for use
from object_detection.utils.config_util import create_pipeline_proto_from_configs
from object_detection.utils.config_util import get_configs_from_pipeline_file
import exporter_tf_serving as exporter

# Configuration for model to be exported
#config_pathname = "faster_rcnn_inception_v2_coco_2018_01_28/pipeline.config" #for faster rcnn inception v2

config_pathname = "D:/tf-serving/faster_rcnn_resnet101_coco_2018_01_28/pipeline.config"

# Input checkpoint for the model to be exported
# Path to the directory which consists of the saved model on disk (see above)
trained_model_dir = "D:/tf-serving/faster_rcnn_resnet101_coco_2018_01_28/"

# Create proto from model confguration
configs = get_configs_from_pipeline_file(config_pathname)
pipeline_proto = create_pipeline_proto_from_configs(configs=configs)

# Read .ckpt and .meta files from model directory
checkpoint = tf.train.get_checkpoint_state(trained_model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

# Model Version
model_version_id = 1

# Output Directory
output_directory = "D:/tf-serving/tf_serving_models/faster_rcnn_resnet101/" + str(model_version_id)

# Export model for serving
exporter.export_inference_graph(input_type='image_tensor',pipeline_config=pipeline_proto,trained_checkpoint_prefix=input_checkpoint,output_directory=output_directory,optimize_graph=False)