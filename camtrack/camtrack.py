#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np

from corners import CornerStorage
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
    compute_reprojection_errors
)
import cv2
from concurrent.futures import ThreadPoolExecutor


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    MAX_REPROJECTION_ERROR = 1.6
    MIN_TRIANGULATION_ANGLE_DEG = 2.39
    MIN_DEPTH = 0.1
    RETRIANGULATION_RANSAC_ITERS = 4
    POSES_RECALC_ITERS = 4
    RETRIANGULATE_ITERS = 5
    MAX_RETRIANGULATES_PER_ITER = 1000
    RETRIANGULATION_FRAME_LIMIT = 25

    retriangulation_params = TriangulationParameters(MAX_REPROJECTION_ERROR, MIN_TRIANGULATION_ANGLE_DEG, MIN_DEPTH)

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    frame_count = len(rgb_sequence)

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

    def populate_cloud(points3d, ids, inliers):
        updated_points = 0
        for i, p, e in zip(ids, points3d, inliers):
            if i not in cloud3d.keys() or e >= cloud3d[i]['inliers']:
                updated_points += 1
                cloud3d[i] = {
                    'pos': p,
                    'inliers': e
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
    populate_cloud(init_p3d, init_ids, 2 * np.ones_like(init_ids))

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

    last_retriangulation = {}

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
            return p3d[0], 2
        if len(frames) > RETRIANGULATION_FRAME_LIMIT:
            order = np.arange(len(frames))
            np.random.shuffle(order)
            order = order[:RETRIANGULATION_FRAME_LIMIT]
            frames, p2d, mats = np.array(frames)[order], np.array(p2d)[order], np.array(mats)[order]
        best_pos = None
        best_inliers = None
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
            if best_pos is None or best_inliers < inliers:
                best_pos = p3d[0]
                best_inliers = inliers
        if best_pos is None:
            return None
        return best_pos, best_inliers

    with ThreadPoolExecutor(max_workers=8) as executor:
        iter_count = 0
        while np.sum([mat is not None for mat in view_mats]) != frame_count:
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
                     if i not in last_retriangulation.keys() or last_retriangulation[
                         i] <= iter_count - RETRIANGULATE_ITERS]
            if len(tasks) > MAX_RETRIANGULATES_PER_ITER:
                np.random.shuffle(tasks)
                tasks = tasks[:MAX_RETRIANGULATES_PER_ITER]
            retr_p3d, retr_ids, retr_inliers = [], [], []
            for i, r in zip(tasks, executor.map(retriangulate, tasks)):
                if r is not None:
                    p3, e = r
                    retr_p3d.append(p3)
                    retr_ids.append(i)
                    retr_inliers.append(e)
                last_retriangulation[i] = iter_count

            updated_points = populate_cloud(retr_p3d, retr_ids, retr_inliers)
            print_info([f'+pose for frame #{best_frame}',
                        f'updated {updated_points} points in the cloud'])

            if iter_count % POSES_RECALC_ITERS == 0:
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
                            f'updated {updated_poses} poses'])

    ids, points = [], []
    for k, v in cloud3d.items():
        ids.append(k)
        points.append(v['pos'])
    point_cloud_builder = PointCloudBuilder(np.array(ids),
                                            np.array(points))

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
