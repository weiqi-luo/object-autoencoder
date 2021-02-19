import imageio
import cv2,os
path = "/home/luo/workspace/object-autoencoder/data/background"
max_count=1000
min_count =  len([name for name in os.listdir(path)])
print(min_count)
try:
    for count in range(min_count,max_count):
        print(count)
        im = imageio.imread("https://picsum.photos/128")
        # cv2.imshow("d",im)
        imageio.imwrite(path+"/{}.jpg".format(count),im)
        count+=1
except KeyboardInterrupt:
    pass
    