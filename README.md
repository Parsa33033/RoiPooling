# RoiPooling
RoiPooling layer of rcnn family. working on both smaller and larger pool size than the region

There are four different conditions that could happend during roi pooling that the code takes care of:
1) if the height and width of the pool is smaller than the height and width of the region
2) if the height and width of the pool is bigger than the height and width of the region
3) if the height of the pool is smaller than the height of the region but the width of the pool is bigger
4) if the height of the pool is bigger than the height of the region but the width of the pool is smaller

you can see conditions number 1 and 2 from the image

![Image](https://github.com/Parsa33033/RoiPooling/blob/master/roipool.png)
