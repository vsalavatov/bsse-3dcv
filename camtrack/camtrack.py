#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import normalize

import sys
import time

from _corners import StorageFilter
from corners import CornerStorage, without_short_tracks
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    eye3x4,
    rodrigues_and_translation_to_view_mat3x4,
    TriangulationParameters,
    Correspondences,
    compute_reprojection_errors,
    _calc_reprojection_error_mask, _calc_z_mask, _calc_triangulation_angle_mask, to_camera_center)
import cv2
from concurrent.futures import ThreadPoolExecutor

CORNER_MIN_FRAMES_COUNT = 12
CORNER_BORDER_THRESHOLD = 40
MAX_REPROJECTION_ERROR = 1.65
MIN_TRIANGULATION_ANGLE_DEG = 2.39
MIN_DEPTH = 0.001
RETRIANGULATION_RANSAC_ITERS = 3
POSES_RECALC_ITERS = 4
MAX_RETRIANGULATIONS_PER_ITER = 500
RETRIANGULATION_FRAME_LIMIT = 30
MIN_COMMON_CORNERS = 11
ESSENTIAL_RANSAC_THRESHOLD = 1.15
MAX_VIEWS_CHECK = 2500

retriangulation_params = TriangulationParameters(MAX_REPROJECTION_ERROR, MIN_TRIANGULATION_ANGLE_DEG, MIN_DEPTH)


def custom_calc_triangulation_angle_mask(view_mat_1: np.ndarray,
                                         view_mat_2: np.ndarray,
                                         points3d: np.ndarray,
                                         min_angle_deg: float):
    camera_center_1 = to_camera_center(view_mat_1)
    camera_center_2 = to_camera_center(view_mat_2)
    vecs_1 = normalize(camera_center_1 - points3d)
    vecs_2 = normalize(camera_center_2 - points3d)
    coss_abs = np.abs(np.einsum('ij,ij->i', vecs_1, vecs_2))
    angles_mask = coss_abs <= np.cos(np.deg2rad(min_angle_deg))
    return angles_mask, coss_abs


def custom_triangulate_correspondences(correspondences: Correspondences,
                                        view_mat_1: np.ndarray, view_mat_2: np.ndarray,
                                        intrinsic_mat: np.ndarray,
                                        parameters: TriangulationParameters):
    points2d_1 = correspondences.points_1
    points2d_2 = correspondences.points_2

    normalized_points2d_1 = cv2.undistortPoints(
        points2d_1.reshape(-1, 1, 2),
        intrinsic_mat,
        np.array([])
    ).reshape(-1, 2)
    normalized_points2d_2 = cv2.undistortPoints(
        points2d_2.reshape(-1, 1, 2),
        intrinsic_mat,
        np.array([])
    ).reshape(-1, 2)

    points3d = cv2.triangulatePoints(view_mat_1, view_mat_2,
                                     normalized_points2d_1.T,
                                     normalized_points2d_2.T)
    points3d = cv2.convertPointsFromHomogeneous(points3d.T).reshape(-1, 3)

    reproj_errs_1 = compute_reprojection_errors(points3d, points2d_1,
                                                intrinsic_mat @ view_mat_1)
    reproj_errs_2 = compute_reprojection_errors(points3d, points2d_2,
                                                intrinsic_mat @ view_mat_2)
    reproj_errs = np.max(np.vstack((reproj_errs_1, reproj_errs_2)), axis=0)
    reproj_error_mask = reproj_errs < parameters.max_reprojection_error

    z_mask = np.logical_and(
        _calc_z_mask(points3d, view_mat_1, parameters.min_depth),
        _calc_z_mask(points3d, view_mat_2, parameters.min_depth)
    )

    angle_mask, cos_angles = custom_calc_triangulation_angle_mask(
        view_mat_1,
        view_mat_2,
        points3d,
        parameters.min_triangulation_angle_deg
    )
    common_mask = reproj_error_mask & z_mask & angle_mask
    count = np.sum(common_mask)
    return points3d[common_mask], correspondences.ids[common_mask], \
           1e9 if count == 0 else np.median(reproj_errs[common_mask]), \
           0 if count == 0 else np.median(cos_angles[common_mask]), \
           reproj_errs[common_mask], cos_angles[common_mask]


def init_views(rgb_sequence, intrinsic_mat, corner_storage):
    print('Searching best initial views...')
    frame_count = len(rgb_sequence)
    with ThreadPoolExecutor() as executor:
        tasks = [(i, j) for i in range(frame_count) for j in range(i + 1, frame_count)]
        torder = np.arange(len(tasks))
        np.random.shuffle(torder)
        tasks = np.array(tasks)[torder[:MAX_VIEWS_CHECK]]

        def try_restore_views(args):
            i, j = args
            corr = build_correspondences(corner_storage[i], corner_storage[j])
            if len(corr.ids) < MIN_COMMON_CORNERS:
                return None
            essential_mat, essential_mask = cv2.findEssentialMat(corr.points_1, corr.points_2,
                                                                 intrinsic_mat,
                                                                 method=cv2.RANSAC, prob=0.999,
                                                                 threshold=ESSENTIAL_RANSAC_THRESHOLD)
            if essential_mat is None:
                return None
            essential_mask = essential_mask.ravel() != 0
            if np.sum(essential_mask) < MIN_COMMON_CORNERS:
                return None

            corr = Correspondences(corr.ids[essential_mask],
                                   corr.points_1[essential_mask],
                                   corr.points_2[essential_mask])

            _, homography_mask = cv2.findHomography(corr.points_1,
                                               corr.points_2,
                                               method=cv2.RANSAC,
                                               ransacReprojThreshold=ESSENTIAL_RANSAC_THRESHOLD,
                                               confidence=0.999,
                                               maxIters=2390)
            h_score = 1.0 - np.sum(homography_mask != 0) / corr.ids.shape[0]

            _, R, t, recover_mask = cv2.recoverPose(essential_mat, corr.points_1, corr.points_2, intrinsic_mat)
            view_mat = np.hstack((R, t))
            pose = view_mat3x4_to_pose(view_mat)
            recover_mask = recover_mask.ravel() != 0
            if np.sum(recover_mask) < MIN_COMMON_CORNERS:
                return None
            corr = Correspondences(corr.ids[recover_mask],
                                   corr.points_1[recover_mask],
                                   corr.points_2[recover_mask])
            _, ids, med_reproj_err, med_cos, _, _ = custom_triangulate_correspondences(
                corr, eye3x4(), view_mat, intrinsic_mat, retriangulation_params
            )
            if len(ids) == 0:
                return None
            return i, j, pose, len(ids), med_reproj_err, med_cos, h_score

        restoration_results = []
        for (ind, i), r in zip(enumerate(tasks), executor.map(try_restore_views, tasks)):
            if ind % 18 == 0:
                print(f'\r{ind / len(tasks)*100:.1f}%', end='')
                sys.stdout.flush()
            if r is not None:
                restoration_results.append(r)
        print()
    assert(len(restoration_results) > 0)

    def score(args):
        _, _, _, inliers, reproj, cos, h_score = args
        return h_score**3.0 * (np.power(inliers / MIN_COMMON_CORNERS, 0.4) - 6 * reproj + 10 * (1 - cos**2))

    restoration_results.sort(key=score, reverse=True)

    print(f'Best frames to begin with: {restoration_results[0][0]} {restoration_results[0][1]} '
          f'(score={score(restoration_results[0]):4f})')
    return restoration_results[0][:3]


def far_from_border(corner_storage: CornerStorage, threshold: int, w, h) -> CornerStorage:
    max_id = max(corners.ids.max() for corners in corner_storage)
    isgood = np.ones((max_id + 1,)).astype(bool)
    for corners in corner_storage:
        for c, id in zip(corners.points, corners.ids):
            y, x = c.ravel()
            if not (threshold <= x <= w - threshold and threshold <= y <= h - threshold):
                isgood[id[0]] = False

    def predicate(corners):
        return isgood[corners.ids.flatten()] == True

    return StorageFilter(corner_storage, predicate)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    corner_storage = without_short_tracks(corner_storage, CORNER_MIN_FRAMES_COUNT)

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    corner_storage = far_from_border(corner_storage, CORNER_BORDER_THRESHOLD, rgb_sequence.frame_shape[0], rgb_sequence.frame_shape[1])
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    frame_count = len(rgb_sequence)

    if known_view_1 is None or known_view_2 is None:
        i, j, pose = init_views(rgb_sequence, intrinsic_mat, corner_storage)
        known_view_1 = (i, view_mat3x4_to_pose(eye3x4()))
        known_view_2 = (j, pose)

    view_mats = [None] * frame_count
    view_mats_inliers = [None] * frame_count
    view_mats[known_view_1[0]], view_mats_inliers[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1]), float('inf')
    view_mats[known_view_2[0]], view_mats_inliers[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1]), float('inf')

    cloud3d = {}
    corner_seen_in_frames = {}
    for i in range(frame_count):
        corners = corner_storage[i]
        for idx, j in enumerate(corners.ids.flatten()):
            if j not in corner_seen_in_frames.keys():
                corner_seen_in_frames[j] = []
            corner_seen_in_frames[j].append((i, idx))

    def populate_cloud(points3d, ids, score):
        updated_points = 0
        for i, p, s in zip(ids, points3d, score):
            if i not in cloud3d.keys() or s > cloud3d[i]['score']:
                updated_points += 1
                cloud3d[i] = {
                    'pos': p,
                    'score': s
                }
        return updated_points

    def triangulate_frame_corrs(frame1, frame2, params=TriangulationParameters(2, 1e-3, 1e-4)):
        corr = build_correspondences(corner_storage[frame1], corner_storage[frame2])
        points3d, ids, median_cos = triangulate_correspondences(corr,
                                                                view_mats[frame1],
                                                                view_mats[frame2],
                                                                intrinsic_mat,
                                                                params)
        return points3d, ids

    init_p3d, init_ids = triangulate_frame_corrs(known_view_1[0], known_view_2[0])
    print(f'Initializing cloud with {len(init_p3d)} points.')
    populate_cloud(init_p3d, init_ids, -100 * np.ones_like(init_ids))

    def print_info(msgs=None):
        # print('\033[K', end='\r') # clear line
        if msgs is None:
            msgs = []
        known_poses_count = np.sum([mat is not None for mat in view_mats])
        print(' | '.join([
                             f'{known_poses_count}/{frame_count} ({known_poses_count / frame_count * 100.0:.0f}%) poses found',
                             f'cloud size: {len(cloud3d)}'
                         ] + msgs))

    def try_restore_pose(frame_id):
        corners = corner_storage[frame_id]
        known_3d_points = []
        known_2d_points = []
        for i, p2d in zip(corners.ids.flatten(), corners.points):
            if i in cloud3d.keys():
                known_3d_points.append(cloud3d[i]['pos'])
                known_2d_points.append(p2d)
        known_3d_points, known_2d_points = np.array(known_3d_points), np.array(known_2d_points)
        if len(known_3d_points) < 4:
            return None
        success, R, t, inliers = cv2.solvePnPRansac(objectPoints=known_3d_points,
                                                    imagePoints=known_2d_points,
                                                    cameraMatrix=intrinsic_mat,
                                                    distCoeffs=None,
                                                    flags=cv2.SOLVEPNP_EPNP,
                                                    confidence=0.9995,
                                                    reprojectionError=MAX_REPROJECTION_ERROR)
        if not success:
            return None

        inliers = np.array(inliers).flatten()
        inliers_count = len(inliers)

        known_3d_points = known_3d_points[inliers]
        known_2d_points = known_2d_points[inliers]
        _, R, t = cv2.solvePnP(objectPoints=known_3d_points,
                               imagePoints=known_2d_points,
                               cameraMatrix=intrinsic_mat,
                               distCoeffs=None,
                               flags=cv2.SOLVEPNP_ITERATIVE,
                               useExtrinsicGuess=True,
                               rvec=R,
                               tvec=t)

        return R, t, inliers_count

    def retriangulate(corner_id):
        frames = []
        p2d = []
        mats = []
        for frame, idx in corner_seen_in_frames[corner_id]:
            if view_mats[frame] is not None:
                frames.append(frame)
                p2d.append(corner_storage[frame].points[idx])
                mats.append(view_mats[frame])
        if len(frames) < 2:
            return None
        if len(frames) == 2:
            p3d, _ = triangulate_frame_corrs(*frames, params=retriangulation_params)
            if len(p3d) == 0:
                return None
            return p3d[0], -100
        if len(frames) > RETRIANGULATION_FRAME_LIMIT:
            order = np.arange(len(frames))
            np.random.shuffle(order)
            order = order[:RETRIANGULATION_FRAME_LIMIT]
            frames, p2d, mats = np.array(frames)[order], np.array(p2d)[order], np.array(mats)[order]
        best_pos = None
        best_score = None

        def score(inliers, med_reproj, cos_angle):
            return np.log(max(1e-4, inliers)) - 3 * med_reproj + 7 * (1 - cos_angle**2)

        for _ in range(RETRIANGULATION_RANSAC_ITERS):
            i, j = np.random.choice(len(frames), 2, replace=False)
            p3d, _, _ = triangulate_correspondences(Correspondences(np.zeros(1),
                                                                    np.array([p2d[i]]),
                                                                    np.array([p2d[j]])),
                                                    mats[i], mats[j], intrinsic_mat,
                                                    parameters=retriangulation_params)
            if len(p3d) == 0:
                continue
            errs = [
                compute_reprojection_errors(p3d, np.array([p2]), intrinsic_mat @ view_mats[f]).flatten()[0]
                for f, p2 in zip(frames, p2d)
            ]
            inliers = np.sum(np.array(errs) <= MAX_REPROJECTION_ERROR)
            _, angle = custom_calc_triangulation_angle_mask(mats[i], mats[j], p3d, 0)
            angle = angle[0]
            s = score(inliers, np.median(errs), angle)
            if best_pos is None or best_score < s:
                best_pos = p3d[0]
                best_score = s
        if best_pos is None:
            return None
        return best_pos, best_score

    try:
        with ThreadPoolExecutor() as executor:
            iter_count = 0
            retriangulation_schedule = dict()
            while np.sum([mat is not None for mat in view_mats]) != frame_count:
                time_begin = time.time()
                iter_count += 1
                tasks = [i for i in range(frame_count) if view_mats[i] is None]
                restoration_results = []

                for i, r in zip(tasks, executor.map(try_restore_pose, tasks)):
                    if r is not None:
                        restoration_results.append((i, r))

                if len(restoration_results) == 0:
                    print('Cannot restore any more camera poses!')
                    break
                best_frame = None
                best_restoration_result = None
                for i, r in restoration_results:
                    if best_restoration_result is None or best_restoration_result[2] < r[2]:  # compare number of inliers
                        best_restoration_result = r
                        best_frame = i

                print_info([f'+pose for frame #{best_frame}',
                            f'inliers count: {best_restoration_result[2]}'])

                view_mats[best_frame] = rodrigues_and_translation_to_view_mat3x4(*best_restoration_result[:2])
                view_mats_inliers[best_frame] = best_restoration_result[2]

                tasks = [i
                         for i in corner_storage[best_frame].ids.flatten()
                         if i not in retriangulation_schedule.keys() or
                            retriangulation_schedule[i][0] <= iter_count]
                if len(tasks) > MAX_RETRIANGULATIONS_PER_ITER:
                    np.random.shuffle(tasks)
                    tasks = tasks[:MAX_RETRIANGULATIONS_PER_ITER]
                retr_p3d, retr_ids, retr_score = [], [], []
                for i, r in zip(tasks, executor.map(retriangulate, tasks)):
                    if r is not None:
                        p3, s = r
                        retr_p3d.append(p3)
                        retr_ids.append(i)
                        retr_score.append(s)
                    if i not in retriangulation_schedule.keys():
                        retriangulation_schedule[i] = [iter_count + 1, -2]
                    else:
                        retriangulation_schedule[i][0] = iter_count + retriangulation_schedule[i][1]
                        retriangulation_schedule[i][1] += 0.67

                updated_points = populate_cloud(retr_p3d, retr_ids, retr_score)
                print_info([f'+pose for frame #{best_frame}',
                            f'updated {updated_points} points in the cloud',
                            f'time elapsed: {time.time() - time_begin:.2f}s'])

                if iter_count % POSES_RECALC_ITERS == 0:
                    time_begin = time.time()
                    tasks = [i for i in range(frame_count) if view_mats[i] is not None]
                    updated_poses = 0
                    for i, r in zip(tasks, executor.map(try_restore_pose, tasks)):
                        if r is not None:
                            R, t, inliers_cnt = r
                            if inliers_cnt >= view_mats_inliers[i]:
                                updated_poses += 1
                                view_mats_inliers[i] = inliers_cnt
                                view_mats[i] = rodrigues_and_translation_to_view_mat3x4(R, t)
                    print_info([f'+pose for frame #{best_frame}',
                                f'updated {updated_poses} poses',
                                f'time elapsed: {time.time() - time_begin:.2f}s'])

                sys.stdout.flush()
    except: # couldn't restore all poses
        pass

    ids, points = [], []
    for k, v in cloud3d.items():
        ids.append(k)
        points.append(v['pos'])
    point_cloud_builder = PointCloudBuilder(np.array(ids),
                                            np.array(points))

    prev_known_view = view_mats[known_view_1[0]]
    for i in range(len(view_mats)):
        if view_mats[i] is None:
            view_mats[i] = prev_known_view
        prev_known_view = view_mats[i]

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        MAX_REPROJECTION_ERROR
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
