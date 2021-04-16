import numpy as np
import cv2
import cv2.aruco as aruco
import pyrealsense2 as rs
import time

def track(matrix_coefficients, distortion_coefficients, frame):

    read = 0
    while read<100:
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale

        #aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)  # Use 5x5 dictionary to find markers

        aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL) 
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters
        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)


        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners, 0.02, matrix_coefficients, distortion_coefficients)
        if tvec is not None:
            print(read)
            print("Position:", tvec[0][0])
            print("Orientation: ", rvec[0][0])
            read += 1
            cv2.imshow("aruco",gray) 
            time.sleep(1)
        if cv2.waitKey(1) == ord('q'):  # q to quit
            #self.cap.release()
            cv2.destroyAllWindows()
            self.pipeline.stop(config)
            raise StopIteration

    return ids, tvec[0][0]

pipeline = rs.pipeline()
config = rs.config()
config.enable_device('048122070949')   # Serial number of D435i
            
# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

print("Jae: Product", device_product_line)
if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
frames = pipeline.wait_for_frames()
color_frame = frames.get_color_frame()
color_intrin = color_frame.profile.as_video_stream_profile().intrinsics    #color_frame.profile.as_video_stream_profile().intrinsics
print(color_intrin)
color_frame = np.asanyarray(color_frame.get_data())
#print(color_np)
mat_coeff = np.array( [[906.692, 0, 647.058], [ 0, 906.353, 363.741], [0, 0, 1]])#435i, 1280x720
distort = np.array( [0, 0, 0, 0, 0])
id,xyz_aruco = track(mat_coeff, distort, color_frame)
self.pipeline.stop(config)    


