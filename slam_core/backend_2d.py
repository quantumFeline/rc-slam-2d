from typing import List

import numpy as np
from scipy import optimize

from slam_core.models import (
    OdometryMeasurement,
    Point2D,
    Pose2D,
    SensorMeasurement,
)

def flatten(item):
    if isinstance(item, (list, tuple)):
        for sub in item:
            yield from flatten(sub)
    else:
        yield item


class SLAM2DBackend:
    """
    The optimization engine.
    """

    def __init__(self):
        pass

    def optimize(
        self,
        initial_poses_guess: List[Pose2D],
        initial_landmarks_guess: List[Point2D],
        measured_movements: List[OdometryMeasurement],
        sensors_measurements: List[SensorMeasurement],
        ground_truth_initial_pose: Pose2D,  # We assume we have the access to the ground truth of starting position
        odometry_noise_std: float = 1.0,
        sensor_noise_std: float = 1.0,
    ) -> optimize.OptimizeResult:
        # Prepare the initial flat parameter vector
        # We optimize over poses and landmarks simultaneously, so they have to be packed into a single, flat numpy array
        # Parameters: [pose_0, pose_1, ..., pose_N, lm1_x, lm1_y, lm2_x, lm2_y, ...]
        ### TODO ###
        print("init poses", initial_poses_guess)
        print("init landmarks", initial_landmarks_guess)
        num_poses = len(initial_poses_guess)
        initial_params = np.array(list(flatten(initial_poses_guess + initial_landmarks_guess)))
        print("initial params", initial_params)
        ### END TODO ###

        # Optimize
        result = optimize.minimize(
            self._objective_function,
            initial_params,
            args=(
                num_poses,
                measured_movements,
                sensors_measurements,
                odometry_noise_std,
                sensor_noise_std,
                ground_truth_initial_pose,
            ),
            method="Powell",
        )

        return result

    def _objective_function(
        self,
        params: np.ndarray,
        num_poses: int,
        measured_movements: List[OdometryMeasurement],
        measurements: List[SensorMeasurement],
        odometry_noise_std: float,
        sensor_noise_std: float,
        ground_truth_initial_pose: Pose2D,
    ) -> float:
        """
        Calculates the error for a given set of parameters (2D pose and landmark estimation).
        """
        # Unpack parameters
        ### TODO ###
        num_pose_params = num_poses * 3
        poses = params[:num_pose_params].reshape(num_pose_params//3, 3) # x, y, angle
        poses_xy = poses[:, 0:2]
        # poses_angle = poses[:, 2]
        landmark_coords = params[num_pose_params:].reshape((len(params)-num_pose_params)//2, 2)
        ### END TODO ###

        # Reconstruct landmarks list [(x,y), ...]
        landmarks = []
        for lm in landmark_coords:
            landmarks.append((lm[0], lm[1]))

        # Avoid division by zero
        odom_weight = (
            1.0 / (odometry_noise_std**2) if odometry_noise_std > 1e-9 else 1.0
        )
        sensor_weight = 1.0 / (sensor_noise_std**2) if sensor_noise_std > 1e-9 else 1.0

        # 1. Movement Penalty (Odometry Error)
        # For each odometry measurement, calculate the expected movement based on the current and next pose, and compare to the measured movement.
        ### TODO ###
        calc_movements = []
        for i in range(len(poses_xy) - 1):
            calc_movements.append(poses_xy[i + 1] - poses_xy[i])

        print("measured movement:", measured_movements[0])
        odometry_measurements = [(measurement.delta_x, measurement.delta_y) for measurement in measured_movements]
        diff_poses = np.array(calc_movements) - np.array(odometry_measurements)
        movement_penalty = np.sum(np.linalg.norm(diff_poses, axis=1)**2) * odom_weight
        # movement_penalty = 0.0
        # for pose_diff in diff_poses:
        #
        #     movement_penalty += np.linalg.norm(pose_diff)**2
        #
        # movement_penalty *= odom_weight

        ### END TODO ###

        # 2. Observation Penalty (Sensor Error)
        # For each pose calculate expected distances and angles to all landmarks and compare to measurements
        ### TODO ###
        distance_penalty = 0
        angle_penalty = 0
        # print("measurements:", measurements)
        # print("poses:", poses_xy)
        # print("landmarks:", landmarks)

        for measurement, pose in zip(measurements, poses_xy):
            # print("measurement:", measurement)
            # print("pose:", pose)
            distances_xy = [(pose - lm) for lm in landmarks]
            # print("distances_xy:", distances_xy)
            expected_distances = [np.linalg.norm(distance) for distance in distances_xy]
            # print("expected distance:", expected_distances)
            distance_penalty += np.linalg.norm(np.array(measurement.distances) - np.array(expected_distances))

            angles = np.array([np.arctan2(distance_y, distance_x) for distance_x, distance_y in distances_xy])
            # print("angles", angles)
            # print("measurement angles", measurement.angles)
            angle_penalty += np.sum(np.abs(angles - measurement.angles))

        distance_penalty *= sensor_weight
        angle_penalty *= sensor_weight
        ### END TODO ###

        # 3. Prior (Anchor first pose)
        # This prevents the whole world from shifting arbitrarily
        # We can weight this heavily to ensure it sticks
        (gt_x, gt_y), gt_theta = ground_truth_initial_pose
        prior = (
            (poses[0][0] - gt_x) ** 2
            + (poses[0][1] - gt_y) ** 2
            + (poses[0][2] - gt_theta) ** 2
        ) * 1000.0

        print("penalties:", movement_penalty, prior, distance_penalty, angle_penalty)
        cost = movement_penalty + prior + distance_penalty + angle_penalty
        return cost
