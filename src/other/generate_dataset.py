import numpy as np
import os

path = "/home/luo/workspace/object-autoencoder/data/object_test"
count = 0

for file in os.listdir(path):
    os.rename(os.path.join(path,file), os.path.join(path,"{}.jpeg".format(count)))
    count += 1

nodata = np.zeros([count,1])
np.save(os.path.join(path,"camera_poses.npy"),nodata)
