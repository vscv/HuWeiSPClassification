# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Simple image classification with Inception.
Run image classification with Inception trained on ImageNet 2012 Challenge data
set.
This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.
Change the --image_file argument to any jpg image to compute a
classification of that image.
Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.
https://tensorflow.org/tutorials/image_recognition/
"""

# ==============================================================================
# 2019_10_03 LSW@NCHC.
#
# Add reading alarm, roi box (x,y) from extra configure file.
# NOTE that, during Popen this RG with parameter, the change happened in ROOT
# "classify.py", who calls RG.exe.
#
# USAGE: time py classify.py --image_file 002051live_201703150917.jpg 2>&-
# ==============================================================================

"""Modified Simple image classification with Inception.
The new Inception-v3 model was retrained on HuWei CCTV dataset.

NOTE: pyinstaller this classify.py to classify.exe before you use it.
"""


import os.path
import re
import sys
import tarfile
import subprocess
import numpy as np
import shlex # for readline list

# pylint: disable=unused-import,g-bad-import-order
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
from six.moves import urllib
# pylint: enable=unused-import,g-bad-import-order


# Get the input image file name
exe_name = sys.argv[0]
in_image = sys.argv[2]
print("This image is:", in_image)
basename = os.path.splitext(in_image)[0]
print(basename)

# Add {cam_roi_file}.cfg to read the (x,y)
cam_roi_file_name = sys.argv[3]
print("Use cam roi from:", cam_roi_file_name)
f = open(cam_roi_file_name, "r")
fp = f.readline()
fp = shlex.split(fp)

# Debug
#fps = [i.split() for i in fp]
#print("cam roi :", fp)
#print("len of fp :",  len(fp))
#print("fp[0]", fp[0])
#print("fp[1]", fp[1])
#print("fp[2]", fp[2])
#print("fp[17]", fp[17])
#exit()

# Set output name
out_infer_name = basename + ".inf"
out_segimg_name = basename + "_" + "seg" + ".jpg"

# Debug
#dist = 1.3
#SegPara = "30 30 120 200 72 150 70 50 70 70 70 90 70 110 70 130 20"
#print(SegPara)
#sys.exit()

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/home/lsw/Images/HuWeiSP/',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """"Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'output_graph_HW.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  if not gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph()

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup()

    # LSW # ADD for save inference reuslt to text file
    text_file = open(out_infer_name,"w")

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
      #text_file.write('%s\n' % (human_string))
      text_file.write('%s (score = %.5f)\n' % (human_string, score))
    text_file.close()

    # LSW # Add parse the inference reuslt is rain or not.
    f_open_infer_file = open(out_infer_name, "r")
    infer_line = f_open_infer_file.readline().splitlines() # readline will add a newline to readed line.
    print("The infer is : ", infer_line, "end")
	
    s1 = infer_line[0][:4]
    s2 = "Rain"
    
    print((str(s1) == "Rain"))
    print("print s1:", s1)
    if (str(s1) == str("Rain")):
      print("Do image segmentation.")
      #p = subprocess.Popen(['/usr/bin/python3.5', 'my_region_growing_cv2.py', in_image, '1.3', '30', '30', '120', '200', '72', '150', '70', '50', '70', '70', '70', '90', '70', '110', '70', '130', '20'], stdout = subprocess.PIPE, stderr=subprocess.PIPE)
      #p = subprocess.Popen(['./RG.exe', in_image, '1.3', '30', '30', '120', '200', '72', '150', '70', '50', '70', '70', '70', '90', '70', '110', '70', '130', '20'], stdout = subprocess.PIPE, stderr=subprocess.PIPE)
      p = subprocess.Popen(['./RG.exe', in_image, fp[0], fp[1] , fp[2], fp[3], fp[4], fp[5],fp[6], fp[7], fp[8], fp[9], fp[10], fp[11], fp[12],fp[13],fp[14],fp[15],fp[16],fp[17]], stdout = subprocess.PIPE, stderr=subprocess.PIPE)
      stdout, stderr = p.communicate()
      print(stdout, stderr)
    else:
      print("No rain go to new image.", infer_line)


def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                             reporthook=_progress)
    print()
    statinfo = os.stat(filepath)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
  #maybe_download_and_extract()
  image = (FLAGS.image_file if FLAGS.image_file else
           os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  run_inference_on_image(image)


if __name__ == '__main__':
  tf.app.run()
