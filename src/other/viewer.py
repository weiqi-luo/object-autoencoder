import math
import trimesh
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer
import numpy as np
from numpy import linalg as LA 
from configparser import ConfigParser
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


def transform_camera(mode):
    """Compute transform matrix for camera in order to pointing towards the object"""
    # sample point for camera position
    # compute euler angle
    t = json.loads(parser.get("cam_pose", "t"))
    print(type(t))

    if not mode:
        print("Using mode 0: derivate rotation matrix from axis angle")
        t = np.asarray(t)
        axisZ = np.array([0,0,1])
        rotAngle = math.acos(np.dot(axisZ,t)/(LA.norm(axisZ)*LA.norm(t)))
        print(rotAngle)
        if rotAngle == 0:
            R = np.identity(3)
        elif rotAngle == 3.141592653589793:
            R = np.array([[math.cos(math.pi), 0, math.sin(math.pi)],
                [0, 1, 0],
                [-math.sin(math.pi), 0, math.cos(math.pi)]])
        else:
            rotVec = np.cross(axisZ,t)
            rotVec = rotVec/LA.norm(rotVec)*rotAngle
            R = Rotation.from_rotvec(rotVec).as_dcm()
            print(R)
        H = np.hstack((R,t.reshape((3,1))))
        H = np.vstack((H,np.array([0,0,0,1])))
        print(H)
        return H

    if mode == 1:
        cam_pose = np.array([json.loads(parser.get("cam_pose", "r1")),
                             json.loads(parser.get("cam_pose", "r2")),
                             json.loads(parser.get("cam_pose", "r3")),
                             json.loads(parser.get("cam_pose", "r4"))])
        print("Using mode 1: directly reading rotation matrix")
        return cam_pose  

    if mode == 2:
        theta_y = json.loads(parser.get("cam_pose", "theta_y"))/180.0*math.pi
        theta_x = json.loads(parser.get("cam_pose", "theta_x"))/180.0*math.pi
        theta_z = json.loads(parser.get("cam_pose", "theta_z"))/180.0*math.pi
        print("Using mode 2: compute rotation matrix from rotation angles")

    if mode == 3:
        if t[1]==0:
            theta_y = math.atan2(t[0],t[2])
        else:
            theta_y = 0
        theta_x = -math.atan2(t[1], t[2])
        # theta_z = -math.atan2(t[0], math.sqrt(t[1]**2+t[2]**2))
        theta_z = -np.sign(t[1])*math.atan2(t[0],abs(t[1]))
        # theta_x = math.asin(t[1] / math.sqrt(t[0]**2 + t[1]**2 + t[2]**2))
        print("Using mode 3: compute rotation matrix according to the position of camera")

    # compute transformation matrix
    Rx = np.array([[1, 0, 0, 0],
                   [0, math.cos(theta_x), -math.sin(theta_x), 0],
                   [0, math.sin(theta_x), math.cos(theta_x), 0],
                   [0, 0, 0, 1]])
    Ry = np.array([[math.cos(theta_y), 0, math.sin(theta_y), 0],
                   [0, 1, 0, 0],
                   [-math.sin(theta_y), 0, math.cos(theta_y), 0],
                   [0, 0, 0, 1]])
    Rz = np.array([[math.cos(theta_z), -math.sin(theta_z), 0, 0],
                   [math.sin(theta_z), math.cos(theta_z), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    T = np.array([[1, 0, 0, t[0]],
                  [0, 1, 0, t[1]],
                  [0, 0, 1, t[2]],
                  [0, 0, 0, 1]])
    print("theta_x=",theta_x/math.pi*180,"theta_y=",theta_y/math.pi*180,"theta_z=",theta_z/math.pi*180)
    print("transformation matrix=", np.linalg.multi_dot([T, Rz, Ry, Rx]) )
    return np.linalg.multi_dot([T, Rz, Ry, Rx])





parser = ConfigParser()
parser.read('../../config/pose.ini')

# Mesh creation
object_trimesh = trimesh.load('../../data/raw/models/002_master_chef_can/textured_simple.obj')
object_mesh = Mesh.from_trimesh(object_trimesh)

# Scene creation
scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

# Adding objects to the scene by manually creating nodes
object_pose = np.array([json.loads(parser.get("obj_pose", "r1")),
                        json.loads(parser.get("obj_pose", "r2")),
                        json.loads(parser.get("obj_pose", "r3")),
                        json.loads(parser.get("obj_pose", "r4"))])
object_node = scene.add(object_mesh, pose=object_pose)
# object_node = Node(mesh=object_mesh, translation=np.array([0.1, 0.15, -np.min(object_trimesh.vertices[:,2])]))
# scene.add_node(object_node)

# Adding lights to the scene by using the add() utility function
light_pose = np.array([json.loads(parser.get("light_pose", "r1")),
                       json.loads(parser.get("light_pose", "r2")),
                       json.loads(parser.get("light_pose", "r3")),
                       json.loads(parser.get("light_pose", "r4"))])
direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                   innerConeAngle=np.pi/16, outerConeAngle=np.pi/6)
point_l = PointLight(color=np.ones(3), intensity=10.0)
direc_l_node = scene.add(direc_l, pose=light_pose)
spot_l_node = scene.add(spot_l, pose=light_pose)

# Adding pre-specified camera to the scene and launch the viewer
cam = PerspectiveCamera(yfov=(np.pi / json.loads(parser.get("cam_pose", "yfov"))))
while True:
    parser.read('../../config/pose.ini')
    cam_pose = transform_camera(json.loads(parser.get("cam_pose", "mode")))
    cam_node = scene.add(cam, pose=cam_pose)
    v = Viewer(scene)
    scene.remove_node(cam_node)

# Rendering offscreen frsom that camera
    # r = OffscreenRenderer(viewport_width=640*2, viewport_height=480*2)
    # color, depth = r.render(scene)
    # r.delete()
    # plt.figure()
    # plt.imshow(color)
    # plt.show()


