from genericpath import exists
import os
import cv2
import tqdm
import struct
import numpy as np
from loguru import logger
from pathlib import Path
from simplejpeg import encode_jpeg
from zod import ZodSequences, constants
from zod.constants import Lidar
from argparse import ArgumentParser
from zod.data_classes.metadata import FrameMetaData, SequenceMetadata
from zod.visualization.lidar_on_image import visualize_lidar_on_image
from mcap_protobuf.writer import Writer as McapWriter
from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
from foxglove_schemas_protobuf.PointCloud_pb2 import PointCloud
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from google.protobuf.timestamp_pb2 import Timestamp
from google.protobuf.wrappers_pb2 import FloatValue, Int64Value, BoolValue, StringValue
from scipy.spatial.transform import Rotation as R


def extrinsics_to_translation_quaternion(matrix):

    assert matrix.shape == (4, 4), "Input must be a 4x4 matrix."

    # Extract translation
    translation = matrix[:3, 3].tolist()

    # Extract rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Convert rotation matrix to quaternion
    rotation = R.from_matrix(rotation_matrix)
    quaternion = rotation.as_quat().tolist()  # Returns [x, y, z, w]

    return quaternion, translation


def lidar_data_to_pointcloud_message(lidar, timestamp, w, t, frame_id="lidar_frame"):

    point_cloud = PointCloud()
    pose = Pose()

    # Set timestamp
    ts = Timestamp()
    ts.FromDatetime(timestamp)
    point_cloud.timestamp.CopyFrom(ts)

    # Set frame ID
    point_cloud.frame_id = frame_id

    # Set pose (set to identity or use actual pose if available)
    pose.position.x = t[0]
    pose.position.y = t[1]
    pose.position.z = t[2]
    pose.orientation.x = w[0]
    pose.orientation.y = w[1]
    pose.orientation.z = w[2]
    pose.orientation.w = w[3]

    # Set point stride and fields
    point_cloud.point_stride = 16  # 4 floats (x, y, z, intensity)
    point_cloud.fields.add(name="x", type="FLOAT32", offset=0)
    point_cloud.fields.add(name="y", type="FLOAT32", offset=4)
    point_cloud.fields.add(name="z", type="FLOAT32", offset=8)
    point_cloud.fields.add(name="intensity", type="FLOAT32", offset=12)

    # Pack point data into bytes
    points = np.asarray(lidar.points, dtype=np.float32)
    intensity = np.asarray(lidar.intensity, dtype=np.float32).reshape([-1, 1])
    lidar_data = np.hstack([points, intensity])

    point_cloud.data = lidar_data.tobytes()

    return point_cloud


def add_sequence_metadata_to_mcap(writer: McapWriter, metadata: SequenceMetadata):

    sequence_id = metadata.sequence_id
    country_code = metadata.country_code
    collection_car = metadata.collection_car
    start_time = metadata.start_time

    ts = Timestamp()
    ts.FromDatetime(start_time)

    logger.info("Adding /sequence/metadata")

    msg = StringValue(value=sequence_id)
    writer.write_message(
        topic="/sequence/metadata/sequence_id",
        log_time=ts.ToNanoseconds(),
        message=msg,
        publish_time=ts.ToNanoseconds(),
    )

    msg = StringValue(value=country_code)
    writer.write_message(
        topic="/sequence/metadata/country_code",
        log_time=ts.ToNanoseconds(),
        message=msg,
        publish_time=ts.ToNanoseconds(),
    )

    msg = StringValue(value=collection_car)
    writer.write_message(
        topic="/sequence/metadata/collection_car",
        log_time=ts.ToNanoseconds(),
        message=msg,
        publish_time=ts.ToNanoseconds(),
    )


def convert_zod_sequences_to_mcap(
    zod_path: str, version: str = "mini", split: str = "val", dest_dir: str = "", skip: bool = False,
):
    """
    Converts ZOD sequences to MCAP format using Foxglove Protobuf schemas (CompressedImage and Pose).
    # TODO
    # Add frame metadata to mcap https://github.com/yaak-ai/zod/blob/main/zod/data_classes/metadata.py#L8
    # Add lidar point cloud as https://github.com/foxglove/schemas/blob/main/schemas/proto/foxglove/PointCloud.proto

    Args:
        zod_path (str): Path to the ZOD sequences dataset.
        output_path (str): Path to the output MCAP file.
    """
    # Load the ZodSequences dataset
    dataset = ZodSequences(dataset_root=zod_path, version=version)  # Adjust as needed
    validation_sequences = dataset.get_split(split)

    for sequence_id in tqdm.tqdm(validation_sequences, ascii=True, unit="seq"):
        sequence = dataset[sequence_id]
        car = sequence.metadata.collection_car

        # Open MCAP file with Protobuf writer
        # s3://yapi-external-datasets/<vendor>/<robot>/<dataset>/<episode_id>/episode.mcap
        mcap_file = Path(
            f"{dest_dir}/{car}/{version}/{split}-{sequence_id}/sequence.mcap"
        )
        mcap_file.parent.mkdir(parents=True, exist_ok=True)

        if mcap_file.exists() and skip:
            logger.info(f"Skipping {mcap_file}, exists")

        logger.info(f"Writing {mcap_file}")
        with open(mcap_file, "wb") as pfile:
            writer = McapWriter(pfile)

            # Sequence metadata
            metadata = sequence.metadata
            add_sequence_metadata_to_mcap(writer, metadata)

            camera_frames = sequence.info.get_camera_frames()
            start_ts = int(sequence.info.start_time.timestamp() * 1e9)
            end_ts = int(sequence.info.end_time.timestamp() * 1e9)

            # GPS data
            # https://github.com/zenseact/zod/blob/main/zod/data_classes/vehicle_data.py#L68
            satellite = sequence.vehicle_data.satellite
            lats = satellite.latpos * 1e-9
            lons = satellite.lonpos * 1e-9
            alts = satellite.altitude
            headings = satellite.heading
            timstamps = satellite.timstamp
            # TODO skip for now
            # speeds = satellite.speeds
            ts = Timestamp()
            logger.info("Adding /sequence/vehicle_data/satellite")
            for lat, lon, alt, heading, ts_nanos in tqdm.tqdm(
                zip(lats, lons, alts, headings, timstamps),
                ascii=True,
                unit="gnss",
                total=len(timstamps),
            ):
                if ts_nanos < start_ts or ts_nanos > end_ts:
                    continue
                ts.FromNanoseconds(ts_nanos)
                gnss = LocationFix(
                    timestamp=ts,
                    frame_id="satellite",
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                )
                writer.write_message(
                    topic="/sequence/vehicle_data/satellite",
                    log_time=ts.ToNanoseconds(),
                    message=gnss,
                    publish_time=ts.ToNanoseconds(),
                )

            # Controls
            # https://github.com/zenseact/zod/blob/main/zod/data_classes/vehicle_data.py#L41
            ego_vehicle_controls = sequence.vehicle_data.ego_vehicle_controls
            acc_pedal = ego_vehicle_controls.acc_pedal
            brake_pedal_pressed = ego_vehicle_controls.brake_pedal_pressed
            steering_angle = ego_vehicle_controls.steering_angle
            steering_angle_rate = ego_vehicle_controls.steering_angle_rate
            steering_wheel_torque = ego_vehicle_controls.steering_wheel_torque
            turn_indicator = ego_vehicle_controls.turn_indicator
            timestamps = ego_vehicle_controls.timestamp
            ts = Timestamp()
            logger.info("Adding /sequence/vehicle_data/ego_vehicle_controls")
            for (
                acc,
                brake,
                steering,
                steering_rate,
                steering_torue,
                turn_signal,
                ts_nanos,
            ) in tqdm.tqdm(
                zip(
                    acc_pedal,
                    brake_pedal_pressed,
                    steering_angle,
                    steering_angle_rate,
                    steering_wheel_torque,
                    turn_indicator,
                    timestamps,
                ),
                ascii=True,
                unit="controls",
                total=len(timestamps),
            ):
                if ts_nanos < start_ts or ts_nanos > end_ts:
                    continue
                msg = FloatValue(value=acc)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_controls/acc_pedal",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = BoolValue(value=brake)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_controls/brake_pedal_pressed",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=steering)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_controls/steering_angle",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=steering_rate)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_controls/steering_angle_rate",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=steering_torue)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_controls/steering_wheel_torque",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = Int64Value(value=turn_signal)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_controls/turn_indicator",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

            # TODO
            # https://github.com/zenseact/zod/blob/main/zod/data_classes/vehicle_data.py#L10
            ego_vehicle_data = sequence.vehicle_data.ego_vehicle_data
            logger.info("Adding /sequence/vehicle_data/ego_vehicle_data")
            roll_rates = ego_vehicle_data.roll_rate
            pitch_rates = ego_vehicle_data.picth_rate
            lat_vels = ego_vehicle_data.lat_vel
            lon_vels = ego_vehicle_data.lon_vel
            lat_accs = ego_vehicle_data.lat_acc
            lon_accs = ego_vehicle_data.lon_acc
            body_heights = ego_vehicle_data.body_height
            body_pitchs = ego_vehicle_data.body_pitch
            timestamps = ego_vehicle_data.timestamp

            for (
                roll_rate,
                pitch_rate,
                lat_vel,
                lon_vel,
                lat_acc,
                lon_acc,
                body_height,
                body_pitch,
                ts_nanos,
            ) in tqdm.tqdm(
                zip(
                    roll_rates,
                    pitch_rates,
                    lat_vels,
                    lon_vels,
                    lat_accs,
                    lon_accs,
                    body_heights,
                    body_pitchs,
                    timestamps,
                ),
                ascii=True,
                unit="state",
                total=len(timestamps),
            ):
                if ts_nanos < start_ts or ts_nanos > end_ts:
                    continue
                msg = FloatValue(value=roll_rate)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_data/roll_rate",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=pitch_rate)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_data/pitch_rate",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=lat_vel)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_data/lat_vel",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=lon_vel)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_data/lon_vel",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=lat_acc)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_data/lat_acc",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=lon_acc)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_data/lon_acc",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=body_height)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_data/body_height",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

                msg = FloatValue(value=body_pitch)
                ts.FromNanoseconds(int(ts_nanos))
                writer.write_message(
                    topic="/sequence/vehicle_data/ego_vehicle_data/body_pitch",
                    log_time=ts.ToNanoseconds(),
                    message=msg,
                    publish_time=ts.ToNanoseconds(),
                )

            # Camera + Lidar data
            logger.info("Adding /sequence/camera/front")
            logger.info(f"Adding /sequence/lidar/{Lidar.VELODYNE.value}")

            for camera_frame in tqdm.tqdm(camera_frames, ascii=True, unit="frame"):
                image = camera_frame.read()
                if image is None:
                    continue
                # Encode image as JPEG
                # image_resize = cv2.resize(
                #     image, dsize=(480, 270), interpolation=cv2.INTER_AREA
                # )
                bytes = encode_jpeg(image)

                # Create CompressedImage message
                ts = Timestamp()
                ts.FromDatetime(camera_frame.time)
                img = CompressedImage(
                    timestamp=ts,
                    format="jpeg",
                    data=bytes,
                )

                writer.write_message(
                    topic="/sequence/camera/front",
                    log_time=ts.ToNanoseconds(),
                    publish_time=ts.ToNanoseconds(),
                    message=img,
                )

                lidar = sequence.info.get_lidar_frame(camera_frame.time).read()
                extrinsics = sequence.calibration.lidars[Lidar.VELODYNE].extrinsics
                w, t = extrinsics_to_translation_quaternion(extrinsics.transform)
                lidar_msg = lidar_data_to_pointcloud_message(
                    lidar, camera_frame.time, w, t
                )

                writer.write_message(
                    topic=f"/sequence/lidar/{Lidar.VELODYNE.value}",
                    log_time=ts.ToNanoseconds(),
                    publish_time=ts.ToNanoseconds(),
                    message=lidar_msg,
                )

            writer.finish()

            logger.info(f"Done writing {mcap_file}")


if __name__ == "__main__":
    # Update these paths to match your local setup

    parser = ArgumentParser("Convert ZOD truckschenes to mcap for Nutron")
    parser.add_argument(
        "-x",
        "--version",
        choices=["full", "mini"],
        help="Which version to convert",
        default="mini",
    )
    parser.add_argument(
        "-s",
        "--split",
        choices=["train", "val"],
        help="Which split to convert",
        default="train",
    )
    parser.add_argument(
        "-r",
        "--root",
        help="ZOD root datapath",
        default="/nasa/3rd_party/zod",
    )
    parser.add_argument(
        "-d",
        "--dest",
        help="Destination mcap dir",
        default="/nasa/3rd_party/zod/mcap",
    )
    parser.add_argument(
        "-e",
        "--skip-existing",
        help="Skip creating mcap file if its exists",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    convert_zod_sequences_to_mcap(
        args.root, args.version, args.split, args.dest, args.skip_existing
    )
