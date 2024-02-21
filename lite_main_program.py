from ultralytics import YOLO
import cv2
from sort import *
from algorithm_analyzer import *

# ==================================================================================
#######################[           Instantiate            ]#########################
# ==================================================================================

# region Instantiate
# Instantiate Required Classes
motion_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# region Colors
colorBlue = (255, 0, 0)
colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)
# endregion

# function to get car ID from plate number and vehicle BBox
def get_car(license_plate, vehicle_IDs):
    """
    Get car bounding box from vehicle detection result list using license plate bounding box

    Args:
        license_plate (np.array[x1, x2, y1, y2]): license plate bounding box
    """

    # boolean variable to flag found statement
    isFound = False

    # Extract variables from license plate
    x1, y1, x2, y2, score, class_id = license_plate

    for i in range(len(vehicle_IDs)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_IDs[i]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_index = i
            isFound = True
            break

    if isFound:
        return vehicle_IDs[car_index] 

    return -1, -1, -1, -1, -1

# function to throw event if any car ID is lost in frames
def tracked_id_checker(frame_loss_tolerance = 30):

    global tracked_IDs
    global vehicle_IDs

    # list containing out of frame car to be processed this frame
    outofframe_vehicle_IDs = []

    # if vehicle_IDs not in tracked_ID, Add the ID Dictionary
    for vehicle_ID in vehicle_IDs:

        # Unpack the Vehicle IDs Data
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_ID

        # Add the ID Dictionary, Reset iteration if the ID is found in vehicles ID
        tracked_IDs[car_id] = 0
    
    # if ID iteration passed through max iteration count, then the event triggered, and delete the data from dictinary
    for tracked_ID in list(tracked_IDs):

        # Increase iteration for each ID on each Frame
        tracked_IDs[tracked_ID] += 1
        
        if tracked_IDs[tracked_ID] > frame_loss_tolerance:
            outofframe_vehicle_IDs.append(tracked_ID)
            del tracked_IDs[tracked_ID]
    
    # GUARD: if out_of_frame vehicle id is empty
    if not outofframe_vehicle_IDs:
        return
    
    # For each frame lost, calculate best result and Update results Dictionary
    if outofframe_vehicle_IDs:
        update_result(outofframe_vehicle_IDs)

# function to update results dictionary
def update_result(outofframe_vehicle_IDs):

    # define counted cars from global variables
    global counted_cars

    # Unpack all data in Result with the Vehicle ID
    for vehicle_ID in outofframe_vehicle_IDs:

        # Increase car counted for every out of frame car
        counted_cars += 1
 
        print(f'CAR COUNTED: {counted_cars} \n LAST CAR ID: {vehicle_ID}')
          
# endregion

# ==================================================================================
#######################[         Global Variables         ]#########################
# ==================================================================================

# region Global Variables
# PATHS
COCO_MODEL_PATH = '../Models/Yolo-Weights/yolov8n.pt'
PLATE_MODEL_PATH = '../Models/Yolo-Weights/surya/best.pt'

# Class ID for Vehicles in Coco dataset
VEHICLE_INDEX = [2, 5, 7]

# Define Maximum temp_results frame capacity
MAX_FRAME_SAVED = 300

# Load Neural Network Model for Vehicle Detection and License Plate Detection
coco_model = YOLO(COCO_MODEL_PATH)
license_plate_detector = YOLO(PLATE_MODEL_PATH)

# Load video sample
cap = cv2.VideoCapture(0) # Used when using video input
# cap = cv2.VideoCapture(0) # Used when using real-time camera

# Global Dictionary to Store Results
temp_results = {} # number of content limited to MAX_FRAME_SAVED
results = {} # Saved to Non-Volatile storage if reached certain number

# Dictionary to store tracked ID for all frame
tracked_IDs = {}

# Car Counter
counted_cars = int(0)
# endregion

# ==================================================================================
#######################[            Main Loop             ]#########################
# ==================================================================================

# region Main Loop
# Read frames
frame_count = -1
ret = True

# Capture frames from video with 60 frameskip
while ret:
    
    frame_count += 1

    ret, frame = cap.read()

    if ret:

        # print Frame count
        #print(f'\n======================= [Processing Frame Number: {frame_count}] =======================')

        # region Frame variables, value cleared and reused every frame

        # Saved a copy of clean frame
        clean_frame = frame.copy()

        # Create dictionary entry using frame number as key
        temp_results[frame_count] = {}

        # Variable to store detection results for vehicles
        detecteds = []

        # Create/Clear Plate number cropped Image List
        cropped_plate_list = []

        vehicle_IDs = []

        # endregion

        #set_start_pin('Frame Process', show_mean=True)

        # Detect vehicles using Yolov8 COCO Trained
        #set_start_pin('Coco Model', show_mean=True)
        detections = coco_model(frame)[0]
        #set_end_pin('Coco Model')
        
        #set_start_pin('Car Loop')
        # Loop for collecting all vehicle result in detecteds list
        for detection in detections.boxes.data.tolist():
            
            # extract bounding boxes from each result
            x1, y1, x2, y2, score, class_id = detection

            # separate bounding box for vehicles detected from all Coco object
            if int(class_id) in VEHICLE_INDEX:

                # Add the vehicle bounding box to the list
                detecteds.append([x1, y1, x2, y2, score])
        #set_end_pin('Car Loop')
        
        # Track Vehicles using SORT
        #set_start_pin('Tracker')
        if detecteds:
            vehicle_IDs = motion_tracker.update(np.asarray(detecteds))
        #set_end_pin('Tracker')

        # detect license plates using License Plate Detection Model
        #set_start_pin('Plate Model', show_mean=True)
        license_plates = license_plate_detector(frame)[0]
        #set_end_pin('Plate Model')
        
        # Loop for collecting all license plate result in license plates list
        for license_plate in license_plates.boxes.data.tolist():

            # extract bounding boxes from each result
            x1, y1, x2, y2, score, class_id = license_plate

            # Skip detected plate if the confidence number is too low
            if score > 0.4:

                # Get Car bounding box for each license plate
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, vehicle_IDs)

                # continue the script only if car is found for the specific license plate
                if (car_id != -1):

                    # Delete earlier frame data if max frame is reached, Significantly affect update_result speed and memory
                    if len(temp_results) > MAX_FRAME_SAVED:
                        del temp_results[min(temp_results.keys())]
                    
                    # Add car bbox to dictionary
                    temp_results[frame_count][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]}}               
        
        tracked_id_checker()

        # region Visualization
        # dupliicate the original frame
        display_frame = frame.copy()
        
        # Draw Bounding boxes for each car in Detected List
        for detection in vehicle_IDs:

            # Extract BBox and Car ID data from Vehicle_ID
            x1, y1, x2, y2, car_id = detection

            # Draw Bounding boxes for each car in Detected List
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)),  colorBlue, 5)

            # Add Car ID to the Car BBox and License plate BBox
            cv2.putText(display_frame, f'{int(car_id)}', (max(0, int(x1)), max(40, int(y1))), cv2.FONT_HERSHEY_PLAIN, 3, colorBlue, 3)

        # Draw bounding boxes for each plate in License_plates List
        for license_plate in license_plates.boxes.data.tolist():

            # extract bounding boxes from each result
            x1, y1, x2, y2, score, class_id = license_plate

            # Draw Bounding boxes for each plates in Detected List
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)),  colorGreen, 5)
        
        # Put Car Counted
        cv2.putText(display_frame, f'Car Counter: {int(counted_cars)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, colorBlue, 3)
        
        # Show the image
        # display_frame = cv2.resize(display_frame, (960, 540), interpolation = cv2.INTER_AREA)
        cv2.imshow('Detection', display_frame)
        cv2.waitKey(1)
        # endregion

    
    else:
        break
    
# releasing the video capture
cap.release()

# endregion

# ==================================================================================
#######################[             TRACING              ]#########################
# ==================================================================================

# region Tracing

# endregion


