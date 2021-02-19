import numpy as np
import imageio
import matplotlib.pyplot as plt
# import visvis as vv
import cv2
import copy

import imgaug as ia
from imgaug import augmenters as iaa
# ori = imageio.imread("test.jpg")
# print(ori)
# im = np.array(ori-255,dtype=bool)



# # # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
# cv2.imshow("image",ori)
# cv2.waitKey(0)

mask = np.array(np.load("test.npy"),dtype=np.float)
print(type(mask[0,0]),np.min(mask),np.max(mask))
print(mask)

ori = np.array(imageio.imread("test.jpg"))
after = np.transpose(ori, axes=(2,0,1))
after = after*mask
after = np.array(np.transpose(after, axes=(1,2,0)),dtype=np.uint8)


seq = iaa.Sequential([        #TODO
    iaa.GaussianBlur(sigma=(0, 0.5))])

after = seq(images=after)

cv2.imshow("image",after)
cv2.waitKey(0)