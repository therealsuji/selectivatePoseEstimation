from scipy import spatial
import addon_utils
import bpy
import bmesh
import math
import numpy as np
import os
import mathutils
import subprocess
import sys
os.system('cls')


def enable_rigify(status=True):
    if(status):
        addon_utils.enable("rigify")
    else:
        addon_utils.disable("rigify")


def clean_rig():
    armature = bpy.data.objects['metarig'].data
    armatureObj = bpy.data.objects['metarig']
    bpy.context.view_layer.objects.active = armatureObj
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    for pb in armature.edit_bones:
        if(pb.name == 'face' or pb.name == 'breast.L' or pb.name == 'breast.R' or pb.name == 'hand.L' or pb.name == 'hand.R'):
            if(len(pb.children) == 0):
                armature.edit_bones.remove(pb)
                continue
            for bone in pb.children_recursive:
                armature.edit_bones.remove(bone)

    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)


def generate_rig():
    bpy.ops.object.armature_human_metarig_add()


def transform_bone():
    armature = bpy.data.objects['metarig']
    poseBones = armature.pose.bones
    for bone in poseBones:
        if(bone.name == "thigh.L"):
            print(bone.rotation_euler[0])
            bone.rotation_mode = "XYZ"
            bone.rotation_euler[0] = 1


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# make sure to run using administrator mode


def install_package(packageName):
    py_exec = bpy.app.binary_path_python
    # ensure pip is installed
    subprocess.call([str(py_exec),        "-m",
                     "ensurepip",        "--user"])
    # update pip
    subprocess.call([str(py_exec),        "-m",        "pip",
                     "install",        "--upgrade",        "pip"])
    # install packages
    subprocess.call([str(py_exec), "-m", "pip", "install",
                     f"--target={str(py_exec)[:-14]}" + "lib", packageName])


# install_package('scipy')


def rotationMatrixToEulerAngles(R):
    assert(isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 /
                                                      np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# enable_rigify()
# generate_rig()
# clean_rig()


# keypoints = np.load("D:\Projects\FYP\selectivatePoseEstimation\outputfile.npy")
# num_frame = len(keypoints)
# # print(keypoints)


# armature = bpy.data.objects['metarig']
# pb = armature.pose.bones.get('thigh.R')
# # pb.location = (0, 0, 1)
# # pb.keyframe_insert("location", frame=2)

# vc = np.array([[2.17057392,  3.02682519,  1.59861243]])
# global_location = armature.matrix_world @ pb.matrix @ pb.location
# pbr = np.array([global_location[:]])

# rot1 = spatial.transform.Rotation.align_vectors(pbr, vc)

# rot = rot1[0].as_euler('xyz', degrees=True)
# pb.rotation_mode = 'XYZ'
# pb.rotation_euler = rot
# #eu = rotationMatrixToEulerAngles(rot)
# # print(eu)
# for pose_bone in armature.pose.bones:
#     x = pose_bone
# #pb = armature.pose.bones.get('upper_arm.L')


def size_bone(point_name1, point_name2, bone):
    p1 = bpy.data.objects[point_name1]
    p2 = bpy.data.objects[point_name2]
    # edit bones
    if bpy.context.active_object.mode == 'EDIT':
        bpy.context.object.data.edit_bones[bone].length = distance(p1, p2)
    else:
        bpy.ops.object.editmode_toggle()
        bpy.context.object.data.edit_bones[bone].length = distance(p1, p2)
    bpy.ops.object.editmode_toggle()

size_bone("Point.008", "Point.001", "Spline")