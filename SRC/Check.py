# ==============================================================================
# 2017_04_15 LSW@NCHC.
#
# Change 3 code to use new in, out dir name for fit the needs.
# cp new.image to /out/ do not need to chnage code of classify.py.
#
# USAGE: time py Check.py /home/TF_io/
# ==============================================================================

"""Daemon function with Popen call.
Glue code to check image dir then call next function.

NOTE: pyinstaller this Check.py to Check.exe before you use it.
"""


import os, time
import sys
from shutil import copyfile
import subprocess


this_n = sys.argv[0]
io_dir = sys.argv[1]
path_to_watch = io_dir + "/in/"
path_to_check = io_dir + "/out/"

before = dict ([(f, None) for f in os.listdir(path_to_watch) if f.endswith('.jpg')])
while 1:
  time.sleep (1)
  after = dict ([(f, None) for f in os.listdir(path_to_watch) if f.endswith('.jpg')])
  for f in after:
    if not f in before:
      #print("Added: ", ", ".join (f))
      # check if cfg exist, else exit this loop
      if os.path.isfile(io_dir + "/" + f.split('_')[0] + ".cfg"):
        print("roi_cfg,", io_dir + "/" + f.split('_')[0] + ".cfg", "exist:", os.path.isfile(io_dir + "/" + f.split('_')[0] + ".cfg"))
        print("New Image Found:", f)
        print("cp",path_to_watch + f, "to", path_to_check + f)
        copyfile(path_to_watch + "/" + f, path_to_check + "/" + f)
        print("roi_cfg:", f.split('_')[0] + ".cfg")
        
        # Call classify.exe
        path_out_img = path_to_check + f
        path_cam_cfg = io_dir + "/" + f.split('_')[0] + ".cfg"
        p = subprocess.Popen(['./classify.exe', "--image_file", path_out_img, path_cam_cfg, "--model_dir", "hw_model"], stdout = subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate() 
        print(stdout, stderr)
      else:
        print("roi_cfg,", io_dir + "/" + f.split('_')[0] + ".cfg", "exist:", os.path.isfile(io_dir + "/" + f.split('_')[0] + ".cfg"))
  removed = [f for f in before if not f in after]
  #if added: 
  #  for a in added:
  #    print("Added: ", ", ".join (a))
  #    print(roi_cfg = a.split('_'))
  if removed: print("Removed: ", ", ".join (removed))
  before = after

