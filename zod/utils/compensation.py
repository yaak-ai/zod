from typing import Union

import numpy as np

from zod.zod_dataclasses.oxts import EgoMotion
from zod.zod_dataclasses.zod_dataclasses import LidarCalibration, LidarData


def motion_compensate_scanwise(
    lidar_data: LidarData,
    ego_motion: EgoMotion,
    calibration: LidarCalibration,
    target_timestamp: float,
) -> LidarData:
    """Motion compensate a (pointwise compensated) lidar point cloud."""
    source_pose = ego_motion.get_poses(lidar_data.core_timestamp)
    target_pose = ego_motion.get_poses(target_timestamp)
    # Compute relative transformation between target pose and source pose
    odometry = np.linalg.inv(target_pose) @ source_pose
    # project to ego vehicle frame using calib
    lidar_data.transform(calibration.extrinsics)
    # project to center frame using odometry
    lidar_data.transform(odometry)
    # project back to lidar frame using calib
    lidar_data.transform(calibration.extrinsics.inverse)
    return lidar_data


def motion_compensate_pointwise(
    lidar_data: LidarData,
    ego_motion: EgoMotion,
    calibration: LidarCalibration,
    target_timestamp: Union[float, np.float64, None] = None,
) -> LidarData:
    """Motion compensate a lidar point cloud in a pointwise manner."""
    lidar_data = lidar_data.copy()
    target_timestamp = target_timestamp or lidar_data.core_timestamp
    # interpolate oxts data for each frame
    point_poses = ego_motion.get_poses(lidar_data.timestamps)
    target_pose = ego_motion.get_poses(target_timestamp)
    # Compute relative transformation between target pose and point poses
    odometry = np.linalg.inv(target_pose) @ point_poses

    # project to ego vehicle frame using calib
    lidar_data.transform(calibration.extrinsics)
    # project to center frame using odometry
    lidar_data.transform(odometry)
    # project back to lidar frame using calib
    lidar_data.transform(calibration.extrinsics.inverse)

    return lidar_data
