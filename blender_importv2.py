import os
import json
import bpy
from bpy import context
import math
import numpy as np
import os.path
import sys
import addon_utils

os.system('cls')
argv = sys.argv
argv = argv[argv.index("--") + 1:]
path = argv[0]

# remove the default cube file added in every scene
if('Cube' in bpy.data.objects):
    bpy.data.objects.remove(bpy.data.objects['Cube'])
   
if(os.path.exists(path) == False):
    raise FileNotFoundError("Could not find the file")
    
data = np.load(path)


def clean_rig():
    armature = bpy.data.objects['metarig'].data
    armatureObj = bpy.data.objects['metarig']
    bpy.context.view_layer.objects.active = armatureObj
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    armature.edit_bones.remove(armature.edit_bones['spine.001'])
    armature.edit_bones.remove(armature.edit_bones['spine.003'])
    armature.edit_bones.remove(armature.edit_bones['spine.005'])

    for pb in armature.edit_bones:
        if(pb.name == 'face' or pb.name == 'breast.L' or pb.name == 'breast.R' or pb.name == 'hand.L' or pb.name == 'hand.R'):
            if(len(pb.children) == 0):
                armature.edit_bones.remove(pb)
                continue
            for bone in pb.children_recursive:
                armature.edit_bones.remove(bone)
    armature.edit_bones.remove(armature.edit_bones['face'])
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

def enable_rigify(status=True):
    if(status):
        addon_utils.enable("rigify")
    else:
        addon_utils.disable("rigify")

def generate_rig():
    bpy.ops.object.armature_human_metarig_add()

enable_rigify()
generate_rig()
clean_rig()



collection = bpy.data.collections.new("Points")
bpy.context.scene.collection.children.link(collection)

layer_collection = bpy.context.view_layer.layer_collection.children[collection.name]
bpy.context.view_layer.active_layer_collection = layer_collection

# create  the coco keypoints 
for point in range(17):
    bpy.ops.mesh.primitive_plane_add(
        enter_editmode=True, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.ops.mesh.merge(type='CENTER')
    bpy.ops.object.editmode_toggle()
    context.active_object.name = 'Point.'+str(1000+point)[1:]

x = 0
y = 1
z = 2

# animate the coco keypoints
for item in range(len(data)):
    for limb in range(len(data[item])):
        bpy.data.objects["Point."+str(1000+limb)
                         [1:]].location[x] = data[item][limb][x]
        bpy.data.objects["Point."+str(1000+limb)
                         [1:]].location[y] = data[item][limb][y]
        bpy.data.objects["Point."+str(1000+limb)
                         [1:]].location[z] = data[item][limb][z]
        bpy.data.objects["Point."+str(1000+limb)[1:]
                         ].keyframe_insert(data_path="location", frame=item)

# get distance between two vectors
def distance(point1, point2) -> float:
    return math.sqrt((point2.location[0] - point1.location[0]) ** 2 + (point2.location[1] - point1.location[1]) ** 2 + (point2.location[2] - point1.location[2]) ** 2)

# size a named bone according to the distance
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

obs = []
for ob in bpy.context.scene.objects:
    if ob.type == 'ARMATURE':
        obs.append(ob)

armature = obs[len(obs)-1].name
obs[len(obs)-1].select_set(True)
view_layer = bpy.context.view_layer
Armature_obj = obs[len(obs)-1]
view_layer.objects.active = Armature_obj

# size_bone("Point.001", "Point.002", "thigh.R")
# change the default size of a few bones
size_bone("Point.000", "Point.007", "spine")
size_bone("Point.007", "Point.008", "spine.002")
size_bone("Point.000", "Point.001", "pelvis.R")
size_bone("Point.000", "Point.004", "pelvis.L")
size_bone("Point.008", "Point.009", "spine.004")
size_bone("Point.009", "Point.010", "spine.006")

bpy.ops.object.mode_set(mode='POSE')

# add the point constraint to the armature bones
actual_bone = 'thigh.R'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.001"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.002"]

actual_bone = 'shin.R'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.002"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.003"]

actual_bone = 'thigh.L'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.004"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.005"]

actual_bone = 'shin.L'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.005"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.006"]

actual_bone = 'pelvis.R'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.000"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.001"]

actual_bone = 'pelvis.L'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.000"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.004"]

actual_bone = 'spine'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.000"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.007"]

actual_bone = 'spine.002'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.007"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.008"]

actual_bone = 'shoulder.R'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.008"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.014"]

actual_bone = 'shoulder.L'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.008"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.011"]

actual_bone = 'upper_arm.R'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.014"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.015"]

actual_bone = 'forearm.R'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.015"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.016"]

actual_bone = 'upper_arm.L'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.011"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.012"]

actual_bone = 'forearm.L'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.012"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.013"]

actual_bone = 'spine.004'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.008"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.009"]

actual_bone = 'spine.006'
obs[len(obs)-1].data.bones.active = obs[len(obs) -
                                        1].pose.bones[actual_bone].bone
obs[len(obs)-1].pose.bones[actual_bone].bone.select = True
bpy.ops.pose.constraint_add(type='COPY_LOCATION')
bpy.context.object.pose.bones[actual_bone].constraints[0].target = bpy.data.objects["Point.009"]
bpy.ops.pose.constraint_add(type='DAMPED_TRACK')
bpy.context.object.pose.bones[actual_bone].constraints[1].target = bpy.data.objects["Point.010"]

bpy.context.scene.frame_end = len(data)
# bake the constraints to the armature 
bpy.ops.nla.bake(frame_start=1, frame_end=len(data), visual_keying=True, clear_constraints=True, clear_parents=True, bake_types={'POSE'})
bpy.ops.object.mode_set(mode='OBJECT')

collection = bpy.data.collections.get('Points')
for obj in collection.objects:
    bpy.data.objects.remove(obj, do_unlink=True)

bpy.data.collections.remove(collection)

desktop = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

# Prints: C:\Users\sdkca\Desktop
print("The Desktop path is: " + desktop)
bpy.ops.wm.save_as_mainfile(filepath=desktop+"\genAnim.blend")
