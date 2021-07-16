#               Read bag from file                ##
#####################################################


import sys

sys.path.append("/usr/local/lib/python3.6/pyrealsense2")
import pyrealsense2 as rs
import numpy as np
import cv2
import argparse


try:

    parser = argparse.ArgumentParser(description="Write camera streams to a bag file")
    parser.add_argument(
        "-s",
        "--serials",
        type=str,
        help="Comma separated serial numbers of cameras to record",
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        help="Output Width",
        default=512
    )
    parser.add_argument(
        "-he",
        "--height",
        type=int,
        help="Output Height",
        default=384
    )
    args = parser.parse_args()

    serials = []
    if args.serials:
        serials = args.serials.split(sep=",")
    else:
        ctx = rs.context()
        devices = ctx.query_devices()
        for device in devices:
            serials.append(device.get_info(rs.camera_info.serial_number))
    resize_width = args.width
    resize_height = args.height
    sn_to_pipeline = {}

    colorizer = rs.colorizer()
    config = rs.config()
    config.enable_stream(rs.stream.color, rs.format.bgr8, 15)
    config.enable_stream(rs.stream.depth, rs.format.z16, 30)
    align_to = rs.stream.color
    align_function = rs.align(align_to)

    for sn in serials:
        config.enable_device(sn)
        pipeline = rs.pipeline()
        pipeline_profile = pipeline.start(config)

        # Print all the streams that are available for streaming
        rs_pipeline_profile = pipeline.get_active_profile()
        for rs_stream_profile in rs_pipeline_profile.get_streams():
            if rs_stream_profile.is_video_stream_profile():
                rs_sp = rs_stream_profile.as_video_stream_profile()
                print(
                    f"{rs_sp.stream_name():<12}{rs_sp.width():>10}x{rs_sp.height()} {rs_sp.fps()}fps {rs_sp.format()}"
                )
            elif rs_stream_profile.is_motion_stream_profile():
                rs_sp = rs_stream_profile.as_motion_stream_profile()
                print(f"{rs_sp.stream_name():<12} {rs_sp.fps()}fps {rs_sp.format()}")

        sn_to_pipeline[sn] = {
            "pipeline": pipeline,
            "pipeline_profile": pipeline_profile
        }

    while True:
        total_frame = None
        for sn, obj in sn_to_pipeline.items():
            frames = obj["pipeline"].wait_for_frames()
            frames = align_function.process(frames.as_frameset())

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())
            color_frame_image = np.asanyarray(color_frame.get_data())
            print(f"depth {depth_color_image.shape} color {color_frame_image.shape}")
            height, width, _ = color_frame_image.shape
            color_frame_image = cv2.putText(
                color_frame_image,
                sn,
                (width - 200, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )
            combined_image = np.vstack(
                    (
                        cv2.resize(depth_color_image, (resize_width, resize_height)),
                        cv2.resize(color_frame_image, (resize_width, resize_height))
                    )
                )
            if total_frame is None:
                total_frame = combined_image
            else:
                total_frame = np.hstack(
                    (
                        combined_image,
                        total_frame,

                    )
                )
        # Render image in opencv window
        cv2.imshow("Stream", total_frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            break

finally:
    pass