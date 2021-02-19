import math
import trimesh, click
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer,IntrinsicsCamera
import numpy as np
from numpy import linalg as LA 
from configparser import ConfigParser
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from PIL import Image
from os import path
import os,sys,cv2, imageio
from tqdm import tqdm,trange,tqdm_notebook


@click.command()
@click.argument('obj')
@click.option('--test', is_flag=True)
# @click.argument('radius', default=None)
# @click.argument('light', default=40)

def main(obj,test):
    light=50
    if test:
        path_root = path.join("../data", "{}_test".format(obj))
    else:
        path_root = path.join("../data", obj)
    # path of 3d mesh
    path_mesh = '../data/raw/models/{}/textured_simple.obj'.format(obj) 
    print("loading 3d mesh from ", path_mesh)
    # path of poses
    path_poses = path.join(path_root, "camera_poses.npy")
    camera_poses = np.load(path_poses)
    print("loading poses from ", path_poses)
    # path to save
    path_save = path.join(path_root, "masks.npy")  
    print("saving the masks to ", path_save)
    
    if obj == "024_bowl" or "037_scissors":
        radius=0.8
    elif obj == "019_pitcher_base":
        radius=1.5
    else:
        print("wrong object")
        sys.exit()

        
    """Render the images"""
    scene = Scene() # create scene
    object_trimesh = trimesh.load(path_mesh) # add object
    object_mesh = Mesh.from_trimesh(object_trimesh)
    scene.add(object_mesh, pose=np.identity(4))
    direc_l = DirectionalLight(color=np.ones(3), intensity=light) # create light
    spot_l = SpotLight(color=np.ones(3), intensity=light,innerConeAngle=np.pi/16.0,outerConeAngle=np.pi/6.0)
    cam = IntrinsicsCamera(fx=570, fy=570, cx=64, cy=64) # create camera
    renderer = OffscreenRenderer(viewport_width=128, viewport_height=128) # create off-renderer

    # Adding pre-specified camera to the scene and launch the viewer
    mask_list = []
    pbar = tqdm(camera_poses)
    for ind, cam_pose in enumerate(pbar):
        cam_node = scene.add(cam, pose=cam_pose)
        # Rendering offscreen frsom that camera
        color, depth = renderer.render(scene)
        mask = np.array(depth,dtype=bool)
        # np.save('test.npy',mask)
        # plt.figure()
        # plt.imshow(mask)
        # plt.show()
        scene.remove_node(cam_node)
        mask_list.append(mask)

        #TODO TEST
        img_color = np.array(imageio.imread(path.join(path_root,"{}.jpeg".format(ind))))
        img_masked = np.transpose(img_color, axes=(2,0,1))
        img_masked = img_masked*mask
        img_masked = np.array(np.transpose(img_masked, axes=(1,2,0)),dtype=np.uint8)
        # cv2.imshow("after",img_masked)
        # cv2.imshow("before",img_color)
        # cv2.waitKey(0)
        
    np.save(path_save,mask_list)

if __name__ == "__main__":
    main()

