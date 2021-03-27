#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf

# Object detection imports
from utils import backbone
from api import object_counting_api

input_image = "./input_images_and_videos/photo_2020-11-15_12-43-37.jpg"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'oid_bbox_trainable_label_map.pbtxt')

is_color_recognition_enabled = 0

result = object_counting_api.single_image_object_counting(input_image, detection_graph, category_index, is_color_recognition_enabled) # targeted objects counting

print (result)
