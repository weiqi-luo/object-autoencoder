from __future__ import print_function, division
import os
from os import path
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage.color import rgb2gray
import sys
import cv2
import imageio, random
import imgaug as ia
from imgaug import augmenters as iaa
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ObjectPoseDataset(Dataset):
    """Face cpose dataset."""

    def __init__(self, npy_file, root_dir, transform, transform_aug=None):
        """
        Args:
            npy_file (string): Path to the npy file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        mask_file = path.join(*npy_file.split("/")[:-1],"masks.npy")
        self.camera_poses = np.load(npy_file)
        self.background_path = "/home/luo/workspace/object-autoencoder/data/background"
        self.background_num = len([name for name in os.listdir(self.background_path)])-1
        self.mask_list = np.load(mask_file)
        self.root_dir = root_dir
        self.transform = transform
        self.transform_aug = transform_aug
        self.transform_aug1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=(0,0), translate=None, scale=(0.5,1), shear=None, resample=False, fillcolor=(0,0,0)),
            transforms.RandomResizedCrop(size = 128, ratio=(1.0,1.0), scale=(0.3,1.0))])
        self.transform_aug2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor()])
        self.seq = iaa.Sequential([        #TODO
            iaa.GaussianBlur(sigma=(0, 2))])

        

    def __len__(self):
        return len(self.camera_poses)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
             
        cpose = self.camera_poses[idx]
        cpose = torch.from_numpy(cpose)
        mask = self.mask_list[idx]

        img_name = os.path.join(self.root_dir,"{}.jpeg".format(idx))
        image_aug = io.imread(img_name)
        image = io.imread(img_name)
        
        image = self.transform(image)

        if self.transform_aug:
            # masking
            image_aug = np.transpose(image_aug, axes=(2,0,1))
            image_aug = image_aug*mask
            image_aug = np.array(np.transpose(image_aug, axes=(1,2,0)),dtype=np.uint8)
            # augment1
            image_aug = self.transform_aug1(image_aug)
            # add background
            image_aug = np.array(image_aug)
            mask_aug = np.array(image_aug,dtype=bool)
            mask_aug = np.logical_not(mask_aug)
            mask_aug = mask_aug[:,:,0]*mask_aug[:,:,1]*mask_aug[:,:,2]
            mask_aug = np.stack((mask_aug,mask_aug,mask_aug),axis=2)
            im = imageio.imread(path.join(self.background_path,"{}.jpg".format(random.randint(0, self.background_num))))
            image_aug[mask_aug] = im[mask_aug]
            # add occlution
            # mask_occ = np.zeros((128,128),dtype=int)
            # start = np.array((random.randint(1,127),random.randint(1,127)))
            # end = start+[random.randint(50,100),random.randint(50,100)]
            # end = [min(end[0],127),min(end[1],127)]
            # print(start,end)
            # print(mask_occ)
            # mask_occ[start[0]:end[0],start[1]:end[1]] = 1
            # mask_occ = np.stack((mask_occ,mask_occ,mask_occ),axis=2)
            # print(np.min(mask_occ),np.max(mask_occ))
            # image_aug[mask_occ] = 0
            # augment 2
            # image_aug = self.seq(images=image_aug)
            image_aug = self.transform_aug2(image_aug)
            
        else:
            image_aug = image     
        sample = {'image': image, 'image_aug' : image_aug, 'cpose': cpose}

        return sample

    def add_background(self):
        pass


class Rescale(object):
    """Rescale the image in a sample to a given size."""
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, image):
        image = transform.resize(image, (self.output_size, self.output_size))
        return image


class RandomCrop(object):
    """Crop randomly the image in a sample."""
    def __init__(self, output_size):
        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, image):        
        h, w = image.shape[:2]
        new_h, new_w = self.output_size, self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # print("top is {0}, left is {1}".format(top, left))
        image = image[top: top + new_h,
                      left: left + new_w]
        return image


class Nomalize(object):
    def __call__(self, image):                
        image = image / 255   
        return image


class ToGrey(object):
    def __call__(self, image):        
        image = rgb2gray(image)
        return image


class ToRGB(object):
    def __call__(self, image):        
        image = np.stack([image,image,image],axis=2)
        return image


class Flatten():
    def __call__(self, image):        
        image = np.reshape(image,(-1,image.shape[0]*image.shape[1]))
        return image


class ExpandDim(object):
    def __call__(self, image):        
        if image.ndim == 2:
            image = np.expand_dims(image, axis=2)
        return image


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):       
        if image.ndim == 3:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
            image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).float()


# class FurtherAugment():
#     def __call__(self,image):
#         # sometimes = lambda aug: iaa.Sometimes(0.5, aug)
#         # apply the following augmenters to most images
#         seq = iaa.Sequential([       
#             iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
#             iaa.GaussianBlur(sigma=(0, 1)),
#         # crop images by -5% to 10% of their height/width
#             # sometimes(iaa.CropAndPad(
#             #     percent=(-0.05, 0.1),pad_mode=ia.ALL,pad_cval=(0, 255)))])#,
#             # sometimes(iaa.Affine(
#             #     scale={(0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
#             #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
#             #     shear=(-16, 16), # shear by -16 to +16 degrees
#             #     order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
#             #     cval=(255), # if mode is constant, use a cval between 0 and 255
#             #     mode=ia.ALL)) , # use any of scikit-image's warping modes (see 2nd image from the top for examples)        
#             # execute 0 to 5 of the following (less important) augmenters per image
#             # don't execute all of them, as that would often be way too strong
#             # iaa.SomeOf((0, 5),[
#             #     sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
#             #     iaa.OneOf([
#             #         iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
#             #         iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
#             #         iaa.MedianBlur(k=(3, 11)), ]), # blur image using local medians with kernel sizes between 2 and 7
#             #     iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
#             #     iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
#             #     # search either for all edges or for directed edges,
#             #     # blend the result with the original image using a blobby mask
#             #     iaa.SimplexNoiseAlpha(iaa.OneOf([
#             #         iaa.EdgeDetect(alpha=(0.5, 1.0)),
#             #         iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),])),
#             # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
#             # iaa.OneOf([
#             #     iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
#             #     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),]),  
#             # iaa.Invert(0.05, per_channel=True), # invert color channels
#             # iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
#             # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
#             # # either change the brightness of the whole image (sometimes
#             # # per channel) or change the brightness of subareas
#             # iaa.OneOf([
#             #     iaa.Multiply((0.5, 1.5), per_channel=0.5),
#             #     iaa.FrequencyNoiseAlpha(
#             #         exponent=(-4, 0),
#             #         first=iaa.Multiply((0.5, 1.5), per_channel=True),
#             #         second=iaa.LinearContrast((0.5, 2.0)))]),
#             # iaa.LinearContrast((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
#             # iaa.Grayscale(alpha=(0.0, 1.0)),
#             # sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
#             # sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
#             # sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))],
#             # random_order=True)],
#             # random_order=True)
#         images_aug = seq(images=image)
#         return images_aug


# ---------------------------------------------------
# ---------------------------------------------------


def show_cpose_batch(sample_batched):
    """Show image with cpose for a batch of samples."""
    images_batch, cpose_batch = sample_batched['image'], sample_batched['cpose']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)
    grid_border_size = 2
    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.title('Batch from dataloader')


def test_objectPoseDataset():
    object_dataset = ObjectPoseDataset(npy_file='../data/interim/camera_poses.npy',
                                        root_dir='../data/interim/')
    fig = plt.figure()
    for i,sample in enumerate(object_dataset):
        print(i, sample['image'].shape, sample['cpose'].shape)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(sample['image'])
        if i == 3:
            break
    plt.show()


def test_transform():
    object_dataset = ObjectPoseDataset(npy_file='../data/interim/camera_poses.npy',
                                        root_dir='../data/interim/')
    scale = Rescale(256)
    crop = RandomCrop(100)
    composed = transforms.Compose([Rescale(256),
                                RandomCrop(224)])
    # Apply each of the above transforms on sample.
    fig = plt.figure()
    sample = object_dataset[11]
    for i, tsfrm in enumerate([scale, crop, composed]):
        transformed_sample = tsfrm(sample)
        ax = plt.subplot(1, 3, i + 1)
        plt.tight_layout()
        ax.set_title(type(tsfrm).__name__)
        plt.imshow(transformed_sample['image'])
        print(transformed_sample['image'].shape)
    plt.show()


def test_transformDataset():
    transformed_dataset = ObjectPoseDataset(npy_file='../data/interim/camera_poses.npy',
                                           root_dir='../data/interim/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()]))
    for i,sample in enumerate(transformed_dataset):
        print(i, sample['image'].size(), sample['cpose'].size())
        if i == 3:
            break
    

def test_dataloader():
    transform = transforms.Compose([Rescale(28), ToGrey(), ToTensor()])
    transformed_dataset = ObjectPoseDataset(npy_file='../data/interim/camera_poses.npy',
                                           root_dir='../data/interim/',
                                           transform=transform)
    dataloader = DataLoader(transformed_dataset, batch_size=5,shuffle=True, num_workers=4)
    plt.figure()
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['cpose'].size())

        # observe 4th batch and stop.
        show_cpose_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        if i_batch == 3:
            break    

    
if __name__ == '__main__':
    # test_objectPoseDataset()
    # test_transform()
    # test_transformDataset()
    test_dataloader()
