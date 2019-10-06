#!/bin/bash

# If MacOS X switch to zsh, shebang no long need.


#002051live
./classify.exe --image_file ../002051live/2017/03/13/002051live_201703131050.jpg hw_confg/cam_roi_001.txt --model_dir hw_model
./classify.exe --image_file ../002051live/2017/03/14/002051live_201703141044.jpg hw_confg/cam_roi_001.txt --model_dir hw_model
./classify.exe --image_file ../002051live/2017/03/15/002051live_201703151152.jpg hw_confg/cam_roi_001.txt --model_dir hw_model
./classify.exe --image_file ../002051live/2017/03/08/002051live_201703080930.jpg hw_confg/cam_roi_001.txt --model_dir hw_model
./classify.exe --image_file ../002051live/2017/03/08/002051live_201703081040.jpg hw_confg/cam_roi_001.txt --model_dir hw_model
./classify.exe --image_file ../002051live/2017/03/08/002051live_201703081440.jpg hw_confg/cam_roi_001.txt --model_dir hw_model

#009052live
#./classify.exe --image_file ../009052live/2017/03/12/009052live_201703121015.jpg cam_roi_001.txt
#./classify.exe --image_file ../009052live/2017/03/13/009052live_201703131310.jpg cam_roi_001.txt
#./classify.exe --image_file ../009052live/2017/03/14/009052live_201703141014.jpg cam_roi_001.txt
#./classify.exe --image_file ../009052live/2017/03/15/009052live_201703151342.jpg cam_roi_001.txt

