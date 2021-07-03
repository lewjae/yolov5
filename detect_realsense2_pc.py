import argparse
import time
from pathlib import Path

import cv2
from numpy.core.fromnumeric import round_
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import pyrealsense2 as rs
import numpy as np

from models.experimental import attempt_load
#from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
	strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from collections import defaultdict
from realsense_device_manager import DeviceManager, post_process_depth_frame
from calibration_kabsch import PoseEstimation
from helper_functions import get_boundary_corners_2D, convert_depth_frame_to_pointcloud, get_clipped_pointcloud, convert_depth_pixel_to_metric_coordinate, get_masked_pointcloud, convert_pointcloud_to_depth
from measurement_task import calculate_boundingbox_points, new_visualise_measurements

# Import for point cloud visulalization
from opencv_pointcloud_viewer import AppState, mouse_cb, project, view, line3d, grid, axes, frustum, pointcloud

class LoadRS:  # capture Realsense stream
	def __init__(self, pipe='rs', img_size=640, stride=32):
		self.img_size = img_size
		self.stride = stride
		
		# Define some constants 
		resolution_width = 640
		resolution_height = 480
		frame_rate = 15 #15  # fps
		dispose_frames_for_stablisation = 30  # frames
		
		chessboard_width = 6 # squares
		chessboard_height = 9 	# squares
		square_size = 0.055 # 0.0253  0.04 # meters  
		
		# Enable the streams from all the intel realsense devices
		rs_config = rs.config()
		rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
		rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
		rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

		# Use the device manager class to enable the devices and get the frames
		self.device_manager = DeviceManager(rs.context(), rs_config)
		self.device_manager.enable_all_devices()
		
		# Allow some frames for the auto-exposure controller to stablise
		for frame in range(dispose_frames_for_stablisation):
			frames = self.device_manager.poll_frames()

		assert( len(self.device_manager._available_devices) > 0 )

		#self.align = rs.align(rs.stream.color)
		"""
		1: Calibration
		Calibrate all the available devices to the world co-ordinates.
		For this purpose, a chessboard printout for use with opencv based calibration process is needed.
			
		"""
		
		# Get the intrinsics of the realsense device 
		intrinsics_devices = self.device_manager.get_device_intrinsics(frames)

		# Set the chessboard parameters for calibration 
		chessboard_params = [chessboard_height, chessboard_width, square_size] 
			
		# Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
		calibrated_device_count = 0
		while calibrated_device_count < len(self.device_manager._available_devices):
			frames = self.device_manager.poll_frames()
			pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
			transformation_result_kabsch  = pose_estimator.perform_pose_estimation()
			object_point = pose_estimator.get_chessboard_corners_in3d()
			calibrated_device_count = 0
			for device in self.device_manager._available_devices:
				if not transformation_result_kabsch[device][0]:
					print("[Jae] device: ",device)
					print("Place the chessboard on the plane where the object needs to be detected..")
				else:
					calibrated_device_count += 1

			# Save the transformation object for all devices in an array to use for measurements
		transformation_devices={}
		chessboard_points_cumulative_3d = np.array([-1,-1,-1]).transpose()
		for device in self.device_manager._available_devices:
			transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
			points3D = object_point[device][2][:,object_point[device][3]]
			points3D = transformation_devices[device].apply_transformation(points3D)
			chessboard_points_cumulative_3d = np.column_stack( (chessboard_points_cumulative_3d,points3D) )

			# Extract the bounds between which the object's dimensions are needed
			# It is necessary for this demo that the object's length and breath is smaller than that of the chessboard
		chessboard_points_cumulative_3d = np.delete(chessboard_points_cumulative_3d, 0, 1)
		self.roi_2D = get_boundary_corners_2D(chessboard_points_cumulative_3d)
		print(self.roi_2D)
		print("Calibration completed... \n\nPlace the box in the field of view of the devices...")
		#print("transformation_devices: ", transformation_devices)
		#print("roi_2D: ",self.roi_2D)           

		# Enable the emitter of the devices
		self.device_manager.enable_emitter(True)

		# Load the JSON settings file in order to enable High Accuracy preset for the realsense
		self.device_manager.load_settings_json("./HighResHighAccuracyPreset.json")

		# Get the extrinsics of the device to be used later
		extrinsics_devices = self.device_manager.get_depth_to_color_extrinsics(frames)

		# Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
		self.calibration_info_devices = defaultdict(list)
		for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
			for key, value in calibration_info.items():
				self.calibration_info_devices[key].append(value)

		# Processing blocks
		self.state = AppState()
		pc = rs.pointcloud()
		decimate = rs.decimation_filter()
		decimate.set_option(rs.option.filter_magnitude, 2 ** self.state.decimate)
		self.colorizer = rs.colorizer()
		cv2.namedWindow(self.state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
		cv2.resizeWindow(self.state.WIN_NAME, resolution_width, resolution_height)
		cv2.setMouseCallback(self.state.WIN_NAME, mouse_cb)
		

	def __iter__(self):
		self.count = -1
		return self

	def __next__(self):
		self.count += 1
		if cv2.waitKey(1) == ord('q'):  # q to quit
			#self.cap.release()
			cv2.destroyAllWindows()
			self.pipeline.stop()
			raise StopIteration
		
			# Read frame
			# Get frameset of color and depth
			#frames = self.pipeline.wait_for_frames()
		frames = self.device_manager.poll_frames()
		
		#frames = self.align.process(frames)
		# Calculate the pointcloud using the depth frames from all the devices
		#point_cloud = calculate_cumulative_pointcloud(frames, self.calibration_info_devices, self.roi_2D)
		#color_images = {}
		img0 = []
		img = []
		sources = []
		depth_images = []
		#depth_colormaps = []
		for (device, frame) in frames.items():
			#color_images[device] = np.asarray(frame[rs.stream.color].get_data())
			color_image = np.asarray(frame[rs.stream.color].get_data())
			filtered_depth_frame = post_process_depth_frame(frame[rs.stream.depth], temporal_smooth_alpha=0.1, temporal_smooth_delta=80)
			filtered_depth_image = np.asarray(filtered_depth_frame.get_data())
			#filtered_depth_frame = np.asarray(frame[rs.stream.depth].get_data())
			#print("[Jae]: color_image",color_image.shape)
			img0.append(color_image)
			#Letter Box, BGR to RGB, to 3x416x416
			#temp = letterbox(color_image, new_shape=self.img_size)[0]
			img.append(letterbox(color_image, new_shape=self.img_size)[0][:,:,::-1].transpose(2,0,1))
			
			depth_images.append(filtered_depth_image)
			
			#depth_colormap = np. asarray(self.colorizer.colorize(filtered_depth_frame).get_data())
			#depth_colormaps.append(depth_colormap)
			
			sources.append(device)


			# Align the depth frame to color frame
			#aligned_frames = self.align.process(frames)

			# Get aligned frames
			#depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
			#color_frame = aligned_frames.get_color_frame()
   
		print(f'D435i {self.count}: ', end='\n')

		# Padded resize
		#img = letterbox(img0, self.img_size, stride=self.stride)[0]
		# Stack
		img = np.stack(img, 0)
		#cv2.imshow("img",img[0])
		#print(img[0][:,:,0])
		# Convert
		#img = img[::-1, :, :].transpose(0,3,1,2)  # BGR to RGB, to 3x416x416
		#temp = img[0][:,:,::-1].transpose(2,0,1)
		#new_img = np.array([temp])
		#print("[Jae] - img.shape after ",img[])
		#print(new_img[0][2,:,:])
		img = np.ascontiguousarray(img)
		#depth_frames = np.ascontiguousarray(depth_frames)

		return sources, img, img0, depth_images, self.calibration_info_devices, self.roi_2D

	def __len__(self):
		return 0



def detect():
	#source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
	source = "rs"
	weights = "yolov5m_0502_best.pt"
	view_img = True
	imgsz = 640
	webcam = True
	depth_scale = 0.001  #D435i
	augment = 'store_true'
	conf_thres = 0.5 
	iou_thres = 0.45 
	classes = [0,1,2,3,4,5,6]
	agnostic_nms = 'store_true'
	device = ''
	depth_threshold = 0.01
	# Initialize
	set_logging()
	device = select_device(device)
	half = device.type != 'cpu'  # half precision only supported on CUDA

	# Load model
	model = attempt_load(weights, map_location=device)  # load FP32 model
	imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
	if half:
		model.half()  # to FP16

	# Second-stage classifier
	classify = False
	if classify:
		modelc = load_classifier(name='resnet101', n=2)  # initialize
		modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

	# Set Dataloader
	cudnn.benchmark = True  # set True to speed up constant image size inference
	dataset = LoadRS(source, img_size=imgsz)

	# Get names and colors
	names = model.module.names if hasattr(model, 'module') else model.names
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

	# Run inference
	t0 = time.time()
	img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
	_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
	
	for path, img, im0s, depth_frames, calibration_info_devices, roi_2d in dataset:
		
		#point_cloud = convert_depth_frame_to_pointcloud(depth_frames,calibration_info_devices[device][1][rs.stream.depth]))
		point_cloud_cumulative = np.array([-1, -1, -1]).transpose()

		img = torch.from_numpy(img).to(device)
		img = img.half() if half else img.float()  # uint8 to fp16/32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0
		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		# Inference
		t1 = time_synchronized()
		pred = model(img, augment=augment)[0]

		# Apply NMS
		#pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
		pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
		t2 = time_synchronized()

		# Apply Classifier
		if classify:
			pred = apply_classifier(pred, modelc, img, im0s)

		#im0h = np.empty((480,640,3),int)
		
		# Process detections
		for i, det in enumerate(pred):  # detections per image

			cam, s, im0, depth_image, frame = path[i], '%g: ' % i, im0s[i].copy(),depth_frames[i].copy(), dataset.count

			s += '%s '%cam
			p = Path(cam)  # to Path
			s += '%gx%g ' % img.shape[2:]  # print string
	
			# Gathered point_cloud from each camera
			point_cloud = convert_depth_frame_to_pointcloud(depth_image, calibration_info_devices[cam][1][rs.stream.depth])
			point_cloud = np.asanyarray(point_cloud)

			#print("R,t: ", cam, calibration_info_devices[cam][0].pose_mat)
			gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
			if len(det):
				# Rescale boxes from img_size to im0 size
				det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
				bbox = det[:, :4].cpu().numpy()
				#print("Jae,  scaled det[:,:4] ", det[:,:4])
				#print("\nJae - bbox ",bbox.shape, bbox)
				# Print results
				for c in det[:, -1].unique():
					n = (det[:, -1] == c).sum()  # detections per class
					s += f'{n} {names[int(c)]}s, '  # add to string
				
				# Write results
				for *xyxy, conf, cls in reversed(det):

					if view_img:  # Add bbox to image

						xyxy_torch = torch.tensor(xyxy).view(1, 4)
						(x1p, y1p, x2p, y2p) = xyxy_torch.view(-1).tolist()
						(xp,yp,wp,hp) = xyxy2xywh(xyxy_torch).view(-1).tolist()
						#(x,y,w,h) = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
						#(x1,y1,x2,y2) = torch.tensor(xyxy).view(1, 4).view(-1).tolist()
						#print('Jae: ',x1p,y1p,x2p,y2p)
						#cv2.rectangle(im0, (int(x1p),int(y1p)), (int(x2p),int(y2p)), (255,0,0), 8)
						# convert image pixel into meters
						z = depth_image[int(yp),int(xp)]*depth_scale  # center
						z1 = depth_image[int(y1p),int(x1p)]*depth_scale
						z2 = depth_image[int(y2p)-1,int(x2p)-1]*depth_scale
						(x,y,z) = rs.rs2_deproject_pixel_to_point(calibration_info_devices[cam][1][rs.stream.color],[xp,yp],z)
						#(x1,y1,z1) = rs.rs2_deproject_pixel_to_point(calibration_info_devices[cam][1][rs.stream.depth],[x1p,y1p],z1)
						#(x2,y2,z2) = rs.rs2_deproject_pixel_to_point(calibration_info_devices[cam][1][rs.stream.depth],[x2p,y2p],z2)	
						#print("x,y,z inimag coord:   ", x,y,z)
						(xx,yy,zz) = convert_depth_pixel_to_metric_coordinate(z,xp,yp,calibration_info_devices[cam][1][rs.stream.color])
						(xx1,yy1,zz1) = convert_depth_pixel_to_metric_coordinate(z1,x1p,y1p,calibration_info_devices[cam][1][rs.stream.color])	
						(xx2,yy2,zz2) = convert_depth_pixel_to_metric_coordinate(z2,x2p,y2p,calibration_info_devices[cam][1][rs.stream.color])
						#print("xx,yy,zz in imag coord: ", xx, yy, zz)	
						#(xx,yy,zz) = rs.rs2_transform_point_to_point(calibration_info_devices[cam][2],[xx,yy,zz])
						#print("xx,yy,zz after: ", xx, yy, zz)				
						#label = f'{names[int(cls)]} {conf:.2f}'
						label = f'{names[int(cls)]} {conf:.2f}, [{xx:0.2f} {yy:0.2f} {zz:0.2f}]m'
						plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
						(xc,yc,zc) = calibration_info_devices[cam][0].apply_transformation(np.array([xx,yy,zz]).reshape(3,1)).tolist()
						#print("JAE: x,y,z ", x, y, z )
						print("x,y,z in checker coord: ", xc, yc, zc)

						bbox = np.array([xx1,xx2, yy1, yy2])
						#print("Jae -bbox:", bbox, xx2-xx1, yy2-yy1)
						print("point_cloud size before", point_cloud.shape)
						point_cloud = get_masked_pointcloud(point_cloud, bbox)
						#depth_masked = np.asarray(convert_pointcloud_to_depth(point_cloud,calibration_info_devices[cam][1][rs.stream.depth]))
						print("point_cloud size after", point_cloud.shape)
			
						#Show depth
						#depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_masked, alpha=0.03), cv2.COLORMAP_JET)
						#cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
						#cv2.imshow('Depth', depth_colormap)
			else:
				print("Nothing detected")
			# Print time (inference + NMS)
			print(f'{s}Done. ({t2 - t1:.3f}s)')

			# Stream results
			cv2.namedWindow("2D BBox: " + str(p), cv2.WINDOW_NORMAL)
			cv2.imshow("2D BBox: " + str(p), im0)

			#print("Jae im0 shape:", im0.shape)
			#im0h = np.hstack((im0h,im0))

			# Get the point cloud in the world-coordinates using the transformation
			point_cloud = calibration_info_devices[cam][0].apply_transformation(point_cloud)

			# Filter the point cloud based on the depth of the object
			# The object placed has its height in the negative direction of z-axis due to the right-hand coordinate system
			point_cloud = get_clipped_pointcloud(point_cloud, roi_2d)
			point_cloud = point_cloud[:,point_cloud[2,:]<-depth_threshold]
			point_cloud_cumulative = np.column_stack( (point_cloud_cumulative, point_cloud ))

		#cv2.namedWindow("2D BBox: ", cv2.WINDOW_NORMAL)
		#cv2.imshow("2D BBox: ", im0h)

		point_cloud_cumulative = np.delete(point_cloud_cumulative, 0, 1)
			
		# Get the bounding box for the pointcloud in image coordinates of the color imager
		bounding_box_points_color_image, length, width, height = calculate_boundingbox_points(point_cloud_cumulative, calibration_info_devices )

		# Draw the bounding box points on the color image and visualise the results
		new_visualise_measurements(im0s,path, bounding_box_points_color_image, length, width, height)
		
	print(f'Done. ({time.time() - t0:.3f}s)')


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
	# Resize and pad image while meeting stride-multiple constraints
	shape = img.shape[:2]  # current shape [height, width]

	if isinstance(new_shape, int):
		new_shape = (new_shape, new_shape)

	# Scale ratio (new / old)
	r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
	if not scaleup:  # only scale down, do not scale up (for better test mAP)
		r = min(r, 1.0)

	# Compute padding
	ratio = r, r  # width, height ratios
	new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
	dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
	if auto:  # minimum rectangle
		dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
	elif scaleFill:  # stretch
		dw, dh = 0.0, 0.0
		new_unpad = (new_shape[1], new_shape[0])
		ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

	dw /= 2  # divide padding into 2 sides
	dh /= 2

	if shape[::-1] != new_unpad:  # resize
		img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
	top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
	left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
	img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
	return img, ratio, (dw, dh)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
	parser.add_argument('--source', type=str, default='rs', help='source')  # file/folder, 0 for webcam
	parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
	parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
	parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--view-img', action='store_true', help='display results')
	parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
	parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
	parser.add_argument('--augment', action='store_true', help='augmented inference')
	parser.add_argument('--update', action='store_true', help='update all models')
	parser.add_argument('--project', default='runs/detect', help='save results to project/name')
	parser.add_argument('--name', default='exp', help='save results to project/name')
	parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
	opt = parser.parse_args()
	print(opt)

	with torch.no_grad():
		if opt.update:  # update all models (to fix SourceChangeWarning)
			for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
				detect()
				strip_optimizer(opt.weights)
		else:
			detect()


