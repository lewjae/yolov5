import argparse
#from home.jlew.git.yolov5.helper_functions import convert_depth_pixel_to_metric_coordinate
import time
from pathlib import Path
from types import FrameType

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
#from opencv_pointcloud_viewer import *

class LoadRS:  # capture Realsense stream
    def __init__(self, pipe='rs', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride
        
        # Define some constants 
        resolution_width = 640
        resolution_height = 480
        frame_rate = 30 #15  # fps
        dispose_frames_for_stablisation = 30  # frames
        
        chessboard_width = 6 # squares
        chessboard_height = 9 	# squares
        square_size = 0.055 # 0.0253  0.04 # meters  
        
        # Enable the streams from all the intel realsense devices
        rs_config = rs.config()
        rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
        rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
        self.align_function = rs.align(rs.stream.color)

        # Use the device manager class to enable the devices and get the frames
        self.device_manager = DeviceManager(rs.context(), rs_config)
        self.mypipelines = self.device_manager.enable_all_devices()
        
        # Allow some frames for the auto-exposure controller to stablise
        for frame in range(dispose_frames_for_stablisation):
            frames = self.device_manager.poll_frames()

        assert( len(self.device_manager._available_devices) > 0 )

        #self.align = rs.align(rs.stream.color)

        # Get the intrinsics of the realsense device 
        intrinsics_devices = self.device_manager.get_device_intrinsics(frames)

        # Enable the emitter of the devices
        self.device_manager.enable_emitter(True)

        # Load the JSON settings file in order to enable High Accuracy preset for the realsense
        self.device_manager.load_settings_json("./HighResHighAccuracyPreset.json")

        # Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
        self.calibration_info_devices = defaultdict(list)
        #for calibration_info in intrinsics_devices:
        for key, value in intrinsics_devices.items():
            self.calibration_info_devices[key].append(value)

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
        
        
        # Calculate the pointcloud using the depth frames from all the devices
        #point_cloud = calculate_cumulative_pointcloud(frames, self.calibration_info_devices, self.roi_2D)
        #color_images = {}
        img0 = []
        img = []
        sources = []
        depth_frames = []
        
        for (device, frame) in self.mypipelines.items():
            print("frame1: ", frame)
            frame = frame["pipeline"]
            #print("frame2: ", frame)
            frame = frame.wait_for_frames()
            #print("frame3: ", frame)
            frame = self.align_function.process(frame.as_frameset())
            #color_images[device] = np.asarray(frame[rs.stream.color].get_data())
            color_image = np.asarray(frame.get_color_frame().get_data())
            #filtered_depth_frame = np.asarray(post_process_depth_frame(frame[rs.stream.depth], temporal_smooth_alpha=0.1, temporal_smooth_delta=80).get_data())
            filtered_depth_frame = np.asarray(frame.get_depth_frame().get_data())
            #print("[Jae]: color_image",color_image.shape)
            img0.append(color_image)
            #Letter Box, BGR to RGB, to 3    
            img.append(letterbox(color_image, new_shape=self.img_size)[0][:,:,::-1].transpose(2,0,1))
            depth_frames.append(filtered_depth_frame)
            sources.append(device)


            # Align the depth frame to color frame
            #aligned_frames = self.align.process(frames)

            # Get aligned frames
            #depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            #color_frame = aligned_frames.get_color_frame()
   
        print(f'D435i {self.count}: ', end='\n')

        # Stack
        img = np.stack(img, 0)
        img = np.ascontiguousarray(img)
        #depth_frames = np.ascontiguousarray(depth_frames)

        return sources, img, img0, depth_frames, self.calibration_info_devices
    
    def __len__(self):
        return 0

# Aggregate for detected items from multiple cameras
# Assuming that each camera has minimu false positives, take union of all detected items and compute maximum
def aggregate(detected_items):

    items_dict = defaultdict(set)
    for _, val in detected_items.items():
        for  k, v in val.items():
            items_dict[k].add(v)	

    agg_dict = {}
    agg_dict = {k : max(items_dict[k]) for k in items_dict}
    return agg_dict

# YoloV5 object detection inferencer
def detect():
    #source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    source = "rs"
    weights = "yolov5m_0502_best.pt"
    view_img = True
    imgsz = 640

    augment = 'store_true'
    conf_thres = 0.75 
    iou_thres = 0.45 
    classes = [0,1,2,3,4,5,6]
    agnostic_nms = 'store_true'
    device = ''
    cut_off_distance = 1.1  #in meter. cuf-off distance for object detection

    depth_scale = 0.001  #D435i
    clipping_distance_in_meters = 0.98 #1.02 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

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
    
    # Initialize cv-detected item dictionary
    detected = {}

    for path, img, im0s, depth_frames, calibration_info_devices in dataset:

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
        gray_color = 155

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            cam, s, im0, depth_image, frame = path[i], '%g: ' % i, im0s[i].copy(),depth_frames[i].copy(), dataset.count

            s += '%s '%cam
            p = Path(cam)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string

            # Initialize the cv-detected item dictionary
            detected[cam] = {}
            #detected[cam] = defaultdict(lambda: 0)
            #print("R,t: ", cam, calibration_info_devices[cam][0].pose_mat)

            

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                #bbox = det[:, :4].cpu().numpy()
                #print("Jae,  scaled det[:,:4] ", det[:,:4])
                #print("\nJae - bbox ",bbox.shape, bbox)
                
                covered_img = im0				
                
                # Write results
                for *xyxy, conf, cls in det:

                    if view_img:  # Add bbox to image

                        xyxy_torch = torch.tensor(xyxy).view(1, 4)
                        (xp,yp,wp,hp) = xyxy2xywh(xyxy_torch).view(-1).tolist()

                        # convert image pixel into meters
                        z = depth_image[int(yp),int(xp)]*depth_scale  # center
                        if z < cut_off_distance:
                            (x,y,z) = rs.rs2_deproject_pixel_to_point(calibration_info_devices[cam][0][rs.stream.color],[xp,yp],z)
                            #print("x,y,z inimag coord:   ", x,y,z)
                            #(xx,yy,zz) = convert_depth_pixel_to_metric_coordinate(z,xp,yp,calibration_info_devices[cam][0][rs.stream.color])
                        
                            #print("xx,yy,zz in imag coord: ", xx, yy, zz)	
                            #(xx,yy,zz) = rs.rs2_transform_point_to_point(calibration_info_devices[cam][2],[xx,yy,zz])
                            #print("xx,yy,zz after: ", xx, yy, zz)				
                            label = f'{names[int(cls)]} {conf:.2f}'
                            #label = f'{names[int(cls)]} {conf:.2f}, [{x:0.2f} {y:0.2f} {z:0.2f}]m'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                            if names[int(cls)] in detected[cam]:
                                detected[cam][names[int(cls)]] += 1
                            else:
                                detected[cam][names[int(cls)]] = 1
                            
                            if i == 0:
                                xywh = xyxy2xywh(torch.tensor(xyxy).view(1, 4)).view(-1).tolist()
                                # convert a list of float to a list of int
                                xywh = list(map(int,xywh))

                                covered_img = cover_detected_items(covered_img, xywh, gray_color)
                                depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channel
                                bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), gray_color, covered_img)
                                cv2.namedWindow('Undetected Items', cv2.WINDOW_NORMAL)
                                cv2.imshow('Undetected Items', bg_removed)
                        
                        else:
                            print("item a far way: ",names[int(cls)], z)

            else:
                if i == 0:
                    depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channel
                    bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), gray_color, im0)
                    cv2.namedWindow('Undetected Items', cv2.WINDOW_NORMAL)
                    cv2.imshow('Undetected Items', bg_removed)
            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            cv2.namedWindow(str(i) + ": " + str(p), cv2.WINDOW_NORMAL)
            cv2.imshow(str(i) + ": " + str(p), im0)
            #cv2.namedWindow('Undetected Items', cv2.WINDOW_NORMAL)
            #cv2.imshow('Undetected Items', bg_removed)
            #depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            #bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, im0)
            #bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), gray_color, covered_img)

            #cv2.namedWindow("covered_img", cv2.WINDOW_NORMAL)
            #cv2.imshow("covered_img", covered_img)


        print("Detected items: ", detected)
        print("Aggregated: ", aggregate(detected))
        print(" ")


    print(f'Done. ({time.time() - t0:.3f}s)')

def cover_detected_items(img, xywh, grey_color):
    (x,y,w,h) = xywh
    covered_img = img.copy()  #(shape h,w,c)
    bbox = np.array([[[grey_color]*3]*w]*h)
    #print("Jae - bbox", bbox.shape)
    covered_img[y-round(h/2-0.1):y+round(h/2+0.1), x-round(w/2-0.1):x+round(w/2+0.1),:] = bbox

    return covered_img


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

    detect()

