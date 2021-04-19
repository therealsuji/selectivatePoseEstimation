# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from tools.inference_tools.camera import *
from tools.inference_tools.model import *
from tools.inference_tools.loss import *
from tools.inference_tools.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from tools.inference_tools.utils import deterministic_random
from tools.inference_tools.custom_dataset import CustomDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
architecture = "3,3,3,3,3"
output_file = "output.mp4"
export_file = "outputfile"
processed_keypoints_dataset = "output/processed_keypoints/keypoints.npz"
checkpoint_file_path = "checkpoint/pretrained_h36m_detectron_coco.bin"
dataset = CustomDataset(processed_keypoints_dataset)
downsample = 1
subjects_unlabeled =""
subjects_test = "S9,S11"
actions = "*"
disable_optimizations = False
dense = False
stride = 1
causal = False
dropout = 0.25
channels = 1024
test_time_augmentation = True 
viz_no_ground_truth = True
viz_bitrate = 3000
viz_limit = -1
viz_downsample = 1
viz_skip = 0

def fetch(subjects, keypoints, action_filter=None, subset=1, parse_3d_poses=True,):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(
                0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


# Evaluate
def evaluate(test_generator, model_pos, joints, return_predictions=False, use_trajectory_model=False):
    with torch.no_grad():
        if not use_trajectory_model:
            model_pos.eval()
        else:
            model_traj.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            if not use_trajectory_model:
                predicted_3d_pos = model_pos(inputs_2d)
            else:
                predicted_3d_pos = model_traj(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                if not use_trajectory_model:
                    predicted_3d_pos[1, :, joints['joints_left'] +
                                     joints['joints_right']] = predicted_3d_pos[1, :, joints['joints_right'] + joints['joints_left']]
                predicted_3d_pos = torch.mean(
                    predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()


def generate_3d_inference(video_path, render):
    print('Preparing data...')
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions' in anim:
                positions_3d = []
                for cam in anim['cameras']:
                    pos_3d = world_to_camera(
                        anim['positions'], R=cam['orientation'], t=cam['translation'])
                    # Remove global offset, but keep trajectory in first position
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    positions_3d.append(pos_3d)
                anim['positions_3d'] = positions_3d

    print('Loading 2D detections...')
    keypoints = np.load(processed_keypoints_dataset, allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(
        keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(
        dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    for subject in dataset.subjects():
        assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(
            subject)
        for action in dataset[subject].keys():
            assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
                action, subject)
            if 'positions_3d' not in dataset[subject][action]:
                continue

            for cam_idx in range(len(keypoints[subject][action])):

                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
                assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

            assert len(keypoints[subject][action]) == len(
                dataset[subject][action]['positions_3d'])

    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                # Normalize camera frame
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(
                    kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    subjects_semi = [] if not subjects_unlabeled else subjects_unlabeled.split(',')
    if not render:
        subjects_test = subjects_test.split(',')
    else:
        subjects_test = [video_path]

    semi_supervised = len(subjects_semi) > 0
    if semi_supervised and not dataset.supports_semi_supervised():
        raise RuntimeError(
            'Semi-supervised training is not implemented for this dataset')

    action_filter = None if actions == '*' else actions.split(',')
    if action_filter is not None:
        print('Selected actions:', action_filter)

    cameras_valid, poses_valid, poses_valid_2d = fetch(
        subjects_test, action_filter=action_filter, keypoints=keypoints)

    filter_widths = [int(x) for x in architecture.split(',')]
    if not disable_optimizations and not dense and stride == 1:
        # Use optimized model for single-frame predictions
        model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                                   filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels)
    else:
        # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
        model_pos_train = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                                        filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels,
                                        dense=dense)

    model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                              filter_widths=filter_widths, causal=causal, dropout=dropout, channels=channels,
                              dense=dense)

    receptive_field = model_pos.receptive_field()
    print('INFO: Receptive field: {} frames'.format(receptive_field))
    pad = (receptive_field - 1) // 2  # Padding on each side
    if causal:
        print('INFO: Using causal convolutions')
        causal_shift = pad
    else:
        causal_shift = 0

    model_params = 0
    for parameter in model_pos.parameters():
        model_params += parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_pos = model_pos.cuda()
        model_pos_train = model_pos_train.cuda()

    chk_filename = checkpoint_file_path
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(
        chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])
    model_traj = None

    test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                        pad=pad, causal_shift=causal_shift, augment=False,
                                        kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    print('INFO: Testing on {} frames'.format(test_generator.num_frames()))

    if render:
        print('Rendering...')

        input_keypoints = keypoints[video_path]['custom'][0].copy()
        ground_truth = None
        gen = UnchunkedGenerator(None, None, [input_keypoints],
                                 pad=pad, causal_shift=causal_shift, augment=test_time_augmentation,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        prediction = evaluate(gen, model_pos=model_pos, return_predictions=True, joints={
            'joints_left': joints_left, 'joints_right': joints_right
        })

        if output_file is not None:
            if ground_truth is not None:
                # Reapply trajectory
                trajectory = ground_truth[:, :1]
                ground_truth[:, 1:] += trajectory
                prediction += trajectory

            # Invert camera transformation
            cam = dataset.cameras()[video_path][0]
            if ground_truth is not None:
                prediction = camera_to_world(
                    prediction, R=cam['orientation'], t=cam['translation'])
                ground_truth = camera_to_world(
                    ground_truth, R=cam['orientation'], t=cam['translation'])
            else:
                # If the ground truth is not available, take the camera extrinsic params from a random subject.
                # They are almost the same, and anyway, we only need this for visualization purposes.
                for subject in dataset.cameras():
                    if 'orientation' in dataset.cameras()[subject][0]:
                        rot = dataset.cameras()[subject][0]['orientation']
                        break
                prediction = camera_to_world(prediction, R=rot, t=0)
                # We don't have the trajectory, but at least we can rebase the height
                prediction[:, :, 2] -= np.min(prediction[:, :, 2])

            anim_output = {'Reconstruction': prediction}
            if ground_truth is not None and not viz_no_ground_truth:
                anim_output['Ground truth'] = ground_truth

            input_keypoints = image_coordinates(
                input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
            print('Exporting joint positions to', export_file)
            # Predictions are in camera space
            np.save(export_file, prediction)
            print(keypoints_metadata)
            from tools.inference_tools.visualization import render_animation
            render_animation(input_keypoints, keypoints_metadata, anim_output,
                             dataset.skeleton(), dataset.fps(
                             ), viz_bitrate, cam['azimuth'], output_file,
                             limit=viz_limit, downsample=viz_downsample, size=6,
                             input_video_path=video_path, viewport=(
                                 cam['res_w'], cam['res_h']),
                             input_video_skip=viz_skip)


# generate3DInference('masked_output.mp4',True)
