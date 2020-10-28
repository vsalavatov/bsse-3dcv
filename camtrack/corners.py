#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    h, w = frame_sequence.frame_shape[:2]
    block_size = round(max(w, h) * 0.009)
    window_size = 3 * block_size
    max_levels = 3
    max_corners = max(300, min(2000, round(w * h / block_size / block_size)))
    quality_level = 0.03
    min_distance = block_size

    corners = []
    radii = []
    corner_ids = []
    id_counter = 0

    prev_frame_pyramid = None

    def make_mask(corners, radii, mask=None):
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:] = 255
        for (x, y), r in zip(corners, radii):
            rx = np.round(x).astype(int)
            ry = np.round(y).astype(int)
            cv2.circle(mask, (rx, ry), r, thickness=-1, color=0)
        return mask

    for frame_index, frame in enumerate(frame_sequence):
        levels, frame_pyramid = cv2.buildOpticalFlowPyramid((frame * 255).astype(np.uint8), (window_size, window_size),
                                                            max_levels, None, False)

        if corners is not None and len(corners) > 0:
            corners, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_pyramid[0], frame_pyramid[0],
                                                                np.asarray(corners, dtype=np.float32), None,
                                                                winSize=(window_size, window_size)
                                                                )
            status, err = status.ravel(), err.ravel()

            reduce = lambda i, corners, radii, ids, status, err: (np.asarray(corners)[i].tolist(),
                                                                  np.asarray(radii)[i].tolist(),
                                                                  np.asarray(ids)[i].tolist(),
                                                                  status[i], err[i])
            corners, radii, corner_ids, status, err = reduce(status == 1,
                                                             corners, radii, corner_ids, status, err)
            corners, radii, corner_ids, status, err = reduce(err < np.mean(err) + 2.39 * np.std(err),
                                                             corners, radii, corner_ids, status, err)

        mask = make_mask(corners, radii)

        for level, frame_level in enumerate(frame_pyramid):
            candidates = cv2.goodFeaturesToTrack(
                frame_level,
                maxCorners=max_corners - len(corners),
                qualityLevel=quality_level,
                minDistance=min_distance,
                blockSize=block_size,
                mask=mask
            )
            if candidates is not None:
                candidates = candidates.reshape(-1, 2).astype(np.float32)
                cur_radius = block_size
                original_radius = cur_radius * 2 ** level
                for (x, y) in candidates:
                    rx, ry = np.round((x, y)).astype(int)
                    orig_x, orig_y = np.array((x, y)) * 2 ** level
                    if mask[ry, rx] != 0:
                        corners.append((orig_x, orig_y))
                        radii.append(original_radius)
                        corner_ids.append(id_counter)
                        id_counter += 1
                        cv2.circle(mask, (rx, ry), cur_radius, thickness=-1, color=0)

            mask = cv2.pyrDown(mask).astype(np.uint8)
            mask[mask <= 200] = 0

        builder.set_corners_at_frame(frame_index,
                                     FrameCorners(np.array(corner_ids), np.array(corners), np.array(radii)))
        prev_frame_pyramid = frame_pyramid


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
