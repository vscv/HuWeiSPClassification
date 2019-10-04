# ==============================================================================
# 2019_10_03 LSW@NCHC.
#
# Add reading alarm, roi box (x,y) from extra configure file.
# NOTE that, during Popen this RG with parameter, the change happened in ROOT
# "classify.py", who calls RG.exe.
#
# USAGE:  time py RG.py 002051live_201703081350.jpg 1.3 30 30 120 200 72 150 70 50 70 70 70 90 70 110 70 130 20
#
# NOTE: 1. if change z to img, it will segmt whole water region not ROI.
#       2. pyinstaller this RG.py to RG.exe before you use it.
# ==============================================================================


#import cv2
#import regiongrowing_np_search as RG
import os
import sys
import numpy as np
from PIL import Image
import imageio # to replace misc.imread.
from scipy import misc
from scipy import ndimage
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

#Check arg
if len(sys.argv) < 2: # if less than 2 means not arg.
    print("no argument")
    sys.exit()

# Reading image
#img =  misc.imread(sys.argv[1], 'True', 'L')
#imgColor =  misc.imread(sys.argv[1])
img =  Image.open(sys.argv[1]).convert('L')
imgColor =  Image.open(sys.argv[1])

# sperate the basename for the name of output file.
basename = os.path.splitext(sys.argv[1])[0]

# Set growing dist and smooth factor.
dist = float(sys.argv[2])
smooth = int(sys.argv[19])

# Smoothing image
img = ndimage.median_filter(img, smooth)

# Simple ROI area
# ROI: if (30 30 120 200)
# ROI: zero posint is top-left, (30,30) (120,200).
ry1 = int(sys.argv[3])
rx1 = int(sys.argv[4])
ry2 = int(sys.argv[5])
rx2 = int(sys.argv[6])

ROI = np.zeros_like(img) # ROI is numpy.ndarray no an image.
ROI[rx1:rx2, ry1:ry2] = img[rx1:rx2, ry1:ry2]


# Initail pyplot plt for plot ROI rectangle
fig = plt.figure()
currentAxis = plt.gca()

# Create eight neighbors for RG.
neighbours = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]

# create bool mask with size of z
mask = np.zeros_like (img)

# Start coordinate
seedX = int(sys.argv[7]) #zero posint is top-left
seedY = int(sys.argv[8])
seedCol = seedY
seedRow = seedX
stack = [(seedCol,seedRow)]

# Create a mask
z=ROI

# Goto Region Growing
while stack:
    x, y = stack.pop()
    mask[x, y] = True
    for dx, dy in neighbours:
        nx, ny = x + dx, y + dy
        if (0 <= nx < z.shape[0] and 0 <= ny < z.shape[1]
            and not mask[nx, ny] and abs(int(z[nx, ny]) - z[x, y]) <= dist): # if change z to img, it will segmt whole water region not ROI.
            stack.append((nx, ny))

# Check Alarm Points is or not covered by mask
a=sys.argv[9:19] # 5 alarm points (x,y) Left-Top corner
alarmPoints = [(int(a[0]),int(a[1])), (int(a[2]),int(a[3])), (int(a[4]),int(a[5])), (int(a[6]),int(a[7])), (int(a[8]),int(a[9]))]
print("alarmPoints", alarmPoints)

# Plot alarm covered by RG region with r>, others with go
i=0
for x,y in alarmPoints:
    print("alarmPts[(x,y)] = ", x, y)#a[x], a[y])
    if mask[y,x] == 1:      # !!!!!here reverse again!!!!!
        i=i+1
        print("mask[x,y]", x, y, "bool = ", mask[y,x], "#count =", i)
        plt.plot(x, y, 'ro')
    else:
        plt.plot(x, y, 'go')
print("Alarm marker(s):", i)

# Debug
#plt.imshow(mask,cmap='gray')
#plt.plot(seedX, seedY, r'*')
#plt.show()

# Debug, show ROI boundary box
currentAxis.add_patch(Rectangle((ry1, rx1), ry2-ry1, rx2-rx1,fill=None, alpha=0.5,color="blue",linewidth=3, linestyle='--')) # Rectangle(xy, width, height, angle=0.0, **kwargs)
print("ROI", ry1, rx1,ry2 , rx2)

# Debug, Show Contour on final image
#plt.contour(mask)

# extracting Contour  data #
cnt = plt.contour(mask, colors='r', linewidths=3,alpha=0.1)
#print(cnt.collections[0].get_paths())
cntP = cnt.collections[0].get_paths()[0]# get the vertices of the contour
v = cntP.vertices
#print(v)
x = v[:,0]
y = v[:,1]
#plt.plot(x,y, 'bo', alpha=0.1)
#cnt_file = open("OutputContourData.txt", "w")
#cnt_file.write(v)
#cnt_file.close()
cntPName = basename + "_" + "cntP" + "_" + ".txt"
np.savetxt(cntPName, v, fmt='%.0f', header=sys.argv[1] + " " + "OutputContourData" + " " + "Alarm markder(s)="+ str(i)) # 1.198499999999999943e+02 1.220000000000000000e+02 need to roud to int by fmt='%.0f'


# Show reuslt
plt.axis('off') # turn the axis off
#plt.imshow(img, cmap="gray")#cmap="gray") There is L:gray channel only, for ndarray process.
plt.imshow(imgColor)# colors RGB for demo
plt.plot(seedX, seedY, '+', color="green",markersize=10,markeredgewidth=2, fillstyle="full", label="Seed")

# Save final_img
final_img_name = basename + "_" + "roi_seg" + "_" + ".png"
plt.savefig(final_img_name)

# Debug
#plt.show()

