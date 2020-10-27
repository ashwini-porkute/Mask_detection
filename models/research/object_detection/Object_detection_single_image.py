######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained classifier to perform object detection.
# It loads the classifier uses it to perform object detection on an image.
# It draws boxes and scores around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import time
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
###PATH_TO_CKPT = "/home/ashwini/TextDetection_MobilenetSSD/models/research/inference_graph/frozen_inference_graph.pb"
PATH_TO_CKPT = "/home/ashwini/edgetpu/detection/inference_check/softnautics_122_op/frozen_inference_graph.pb"

# Path to label map file
PATH_TO_LABELS = "/home/ashwini/TextDetection_MobilenetSSD/models/research/training/text_label_map.pbtxt"

# Path to image
PATH_TO_IMAGE = "/home/ashwini/Downloads/10_images/HF_DSC02923.JPG"

# Number of classes the object detector can identify
NUM_CLASSES = 1

# Load the label map.
# Label maps map indices to category names, so that when our convolution
# network predicts `5`, we know that this corresponds to `king`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input

start = time.time()

(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

end = time.time()

print("total time taken for {} using .pb file is {} seconds...!!!\n".format(PATH_TO_IMAGE.split('/')[-1],(end - start)))
# Draw the results of the detection (aka 'visulaize the results')
det_threshold = 0.5
indexes= []
bboxes = []
scores_list=[]
im_width = image.shape[1]
im_height = image.shape[0]

for i in range(len(scores[0])):
  if scores[0][i] >= det_threshold:
    indexes.append(i)
    scores_list.append(scores[0][i])

for i in indexes:
  bboxes.append(boxes[0][i].tolist())

  #print("\n")
lines1 = []
lines2 = []
total_lines = []
for i in range(len(scores_list)) :
  line = "text" + " " + str(scores_list[i]) + " "
  lines1.append(line)
#  print("lines 1 ----> ",lines1)
c = 0
for bbox in bboxes:
  text_put = ""
  text_put = text_put + "text" + str(scores_list[c])
  c += 1
  xmin = int(bbox[1]* im_width)
  ymin = int(bbox[0]* im_height)
  xmax = int(bbox[3]* im_width)
  ymax = int(bbox[2]* im_height)
  line = str(xmin) + " " +  str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n"
  lines2.append(line)

  cv2.rectangle(image, (xmin,ymin), (xmax,ymax),(255,255,0),2)
  cv2.putText(image, text_put, (int(bbox[1]* im_width), int(bbox[0]* im_height - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
#print("lines 2 ----> ",lines2)

"""
vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.70)
"""
# All the results have been drawn on image. Now display the image.
cv2.imshow('Object detector', image)
cv2.imwrite("/home/ashwini/Downloads/compare/oldpb_oldtflite/TEXT_DETECTION_RESULT_Ro_90_n_HSV_DSC02587_PB.JPG",image)
# Press any key to close the image
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()

