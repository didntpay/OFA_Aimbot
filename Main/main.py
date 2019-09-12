import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import time
import directKeys as keys
import Mode
import keyboard
import win32gui
import pyautogui
from PIL import ImageGrab
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[17]:


# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')



detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
keys = keys.Keys(None)
class Mode():
  DEBUG = 1
  ANALYSIS = 2
  CLIENT = 3

currentmode = Mode.CLIENT
GRABED_WIDTH = 920
GRABED_HEIGHT = 700
GRABED_AREA = GRABED_HEIGHT * GRABED_WIDTH
ONEMETER_RATIO = 0.0364
#y offset
OFFSET_HEAD = -160

def sendMouse(dx, dy, button):
  availableButton = {
  'left_mouse' : None,
  'mouse_move' : keys.mouse_move,
  'right_mouse' : None,
  }
  button = availableButton.get(button)
  if(button == None):
    raise Exception("This operation is yet available, try another one")
  keys.keys_worker.sendMouse(int(dx), int(dy), button)

def __sendMouse_smooth(dx, dy, button):
  for i in range(50):
    dx_smooth = dx / 50
    dy_smooth = dy / 50
    sendMouse(dx_smooth, dy_smooth, button)

#determining the mid point in percentage
def determineMidPoint(rect): #rect[0]:x1, rect[1] y1
  mid_x = rect[1] + ((rect[3] - rect[1])) / 2
  mid_y = rect[0] + ((rect[2] - rect[0])) / 2
  return (mid_x, mid_y)

def __determineDistance(rect):
  detected_area_ratio = (rect[2] - rect[0]) * (rect[3] - rect[1]) 
  return detected_area_ratio / ONEMETER_RATIO


def findCloestEnemy(boxes, scores, classes):
  target_rect = None
  for i, b in enumerate(boxes[0]):
    #ymin, xmin, ymax, xmax = b
    min_distance = 999 #meter               
    typeDict = category_index[classes[0][i]]
    if typeDict['name'] == 'person' and scores[0][i] > 0.50:
      distance = __determineDistance(b)
      if(distance < min_distance):
        target_rect = boxes[0][i]

  return target_rect

def findDxDy(target_point, crosshair_point = [0.5, 0.5]):
  dx = (target_point[0] - crosshair_point[0]) / 0.5
  dy = (target_point[1] - crosshair_point[1]) / 0.5

  return [dx * GRABED_WIDTH, dy * GRABED_HEIGHT]


with detection_graph.as_default():
  with tf.compat.v1.Session() as sess:
    file = open('object_detection_log_debug.txt', 'w')
    while(True):
      if keyboard.is_pressed('alt'):
        before = time.time()
        #bbox : (left_x, top_y, right_x, bottom_y)
        image_np = np.array(ImageGrab.grab(bbox = (0, 40, 1020, 800)))
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        #image_np = cv2.resize(image_np, (410, 300))
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        boxes, scores, classes, num_detections = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        #file.write("Seconds took in analysis " + str(time.time() - before) + "\n")
        before = time.time()
        # Visualization of the results of a detection.
        
        if currentmode == Mode.CLIENT:
          vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
          target_rect = findCloestEnemy(boxes, scores, classes)
          if not str(target_rect) == 'None':
            points = determineMidPoint(target_rect) # in percentage of the screen
            cv2.circle(image_np, (int(points[0] * GRABED_WIDTH), int(points[1] * GRABED_HEIGHT)), 10, (255, 0, 0), 3)
            #points_converted = [int(points[0]), int(points[1])]
            cursor_pos = win32gui.GetCursorPos()
            #flags, hcursor, screen_mid = win32gui.GetCursorInfo()
            theta_D = findDxDy(points)
            file.write("The mouse location is "  + "\n")
            file.write("The calculated mid point is " + str(points) + "\n")
            file.write("The dx and dy is " + str(theta_D) + "\n")
            sendMouse(theta_D[0], theta_D[1], 'mouse_move')

        elif currentmode == Mode.ANALYSIS:
          vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
          for i, b in enumerate(boxes[0]):
            #ymin, xmin, ymax, xmax = b
            min_distance = 999 #meter               
            typeDict = category_index[classes[0][i]]
            if typeDict['name'] == 'person' and scores[0][i] > 0.65:
              ratio = (b[2] - b[0]) * (b[3] - b[1])
              file.write("Ratio of target at 1 meter is " + str(ratio) + "\n")
          #one_meter_target_rect = findCloestEnemy(boxes, scores, classes)

                  
        #file.write(str(points[0] * GRABED_WIDTH) + " " + str(points[1] * GRABED_HEIGHT) + "\n")
        #

        cv2.imshow("detected", image_np)
      if cv2.waitKey(10) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        file.close()
        break






