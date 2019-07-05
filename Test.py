
from RoiPooling import RoiPooling
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

mode = 'th'
h = 50
w = 38
if mode == 'tf':
    feature_map = np.zeros((h,w,512))
elif mode == 'th':
    feature_map = np.zeros((512,h,w))
for i in range(h):
    for j in range(w):
        if np.random.rand() < 0.1:
            if mode == 'tf':
                feature_map[i,j,:] = np.random.rand()
            elif mode == 'th':
                feature_map[:,i,j] = np.random.rand()

roi_batch = np.array([[0,0,10,10],[2,2,5,5]])

print(feature_map.shape)
roi_pooled = RoiPooling(mode=mode).get_pooled_rois(feature_map, roi_batch)
print(roi_pooled.shape)

# region = RoiPooling(mode=mode).get_region(feature_map, roi_batch[0])

if mode=='tf':
    _, ax = plt.subplots(2)
    ax[0].imshow(feature_map[...,0])
    xmin, ymin, xmax, ymax = roi_batch[0]
    ax[0].add_patch(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none', linewidth=1))
    ax[1].imshow(roi_pooled[0,...,0])
    plt.show()
elif mode=='th':
    _, ax = plt.subplots(2)
    ax[0].imshow(feature_map[0,...])
    xmin, ymin, xmax, ymax = roi_batch[0]
    ax[0].add_patch(patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor='none', linewidth=1))
    ax[1].imshow(roi_pooled[0,0,...])
    plt.show()


