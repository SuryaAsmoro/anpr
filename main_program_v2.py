from ultralytics import YOLO
import cv2
from sort import *
from image_enhancing import *
from ocr_module import *
from data_storage import *
from algorithm_analyzer import *
from difflib import SequenceMatcher
from colorDetection import *
# import tracemalloc

# ==================================================================================
#######################[           Instantiate            ]#########################
# ==================================================================================

# region Instantiate
set_start_pin('Total Process')
# Instantiate Required Classes
motion_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
img_enhancer = EnhancingModule()

# function to get car ID from plate number and vehicle BBox
def get_car(license_plate, vehicle_IDs):

    # boolean variable to flag found statement
    isFound = False

    # Extract variables from license plate
    x1, y1, x2, y2, score = license_plate

    for i in range(len(vehicle_IDs)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_IDs[i]

        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_index = i
            isFound = True
            break

    if isFound:
        return vehicle_IDs[car_index] 

    return -1, -1, -1, -1, -1

# Function to check similarity between 2 string
def similarity_ratio(string_a, string_b):
    return SequenceMatcher(None, string_a, string_b).ratio()

def tracked_id_checker(tracked_IDs, vehicle_IDs, temp_results, frame_loss_tolerance = 3):

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
        update_result(temp_results, outofframe_vehicle_IDs)

def update_result(temp_results, outofframe_vehicle_IDs, merging_range=2):
    
    # Unpack all data in Result with the Vehicle ID
    for vehicle_ID in outofframe_vehicle_IDs:

        # Temporary score and frame number to sort the Highest Score 
        best_text_score = 0
        best_text = ''
        best_color = ''

        for frame_nmr in temp_results.keys():
            if vehicle_ID in temp_results[frame_nmr]:
                if 'car' in temp_results[frame_nmr][vehicle_ID].keys() and \
                    'license_plate' in temp_results[frame_nmr][vehicle_ID].keys() and \
                    'text' in temp_results[frame_nmr][vehicle_ID]['license_plate'].keys():

                    # Get data with Highest Text Score
                    if temp_results[frame_nmr][vehicle_ID]['license_plate']['text_score'] > best_text_score:
                        best_text_score = temp_results[frame_nmr][vehicle_ID]['license_plate']['text_score']
                        best_text = temp_results[frame_nmr][vehicle_ID]['license_plate']['text']
                        best_color = temp_results[frame_nmr][vehicle_ID]['car']['color']

                    # Deleted specific data in temp_result
                    del temp_results[frame_nmr][vehicle_ID]

        # GUARD to skip process if best text = null
        if best_text != '' and best_color != '':
            
            # GUARD to prevent similarity check if the results is still empty
            if results.keys():
                
                # GUARD to prevent list index out of range
                if len(results.keys()) < merging_range:
                    merging_range = len(results.keys())

                i = 1
                while i <= merging_range:

                    key = list(results.keys())[-i]

                    # Check if there are similarity with adjacent data
                    if similarity_ratio(best_text, results[key]['text']) > 0.85:
                        
                        # if current score is higher, delete previous data
                        if best_text_score > results[key]['text_score']:
                            
                            # delete previous data
                            del results[key]

                            # continue to check similarity in other entries
                            continue
                        
                        # if current score is lower
                        else:

                            # drop current data
                            break

                    # if reach limit merging range, 
                    if i >= merging_range:

                        # add current data        
                        results[vehicle_ID] = {'text': best_text, 'text_score': best_text_score, 'color': best_color}

                    # Limit iteration by Merging_range
                    i += 1

            else:
                # add current data
                results[vehicle_ID] = {'text': best_text, 'text_score': best_text_score, 'color': best_color}

# Run tremalloc to Analyze memory allocation
# tracemalloc.start()

# endregion

# ==================================================================================
#######################[         Global Variables         ]#########################
# ==================================================================================

# region Global Variables
# PATHS
INPUT_VIDEO_PATH = '../Data/Videos/margonda5/data_t.mp4'
# MODEL_PATH = '../Models/Yolo-Weights/surya/new/v5/best.pt'
MODEL_PATH = '../Models/Yolo-Weights/surya/new/Evaluated/best.pt'
OUTPUT_PATH = '../Data/playground'

# Class IDs
plate_class_id = 1
vehicle_class_id = 0

# frameskip in frame, Need Work, High Execution Time
FRAMESKIP = 0

# Frame limit (used in test Mode), use -1 for maximum available
MAX_FRAME = -1     

# Starting frame (used in test Mode), use 0 to start from begining
STARTING_FRAME = 0

# Define Maximum temp_results frame capacity, Significantly affect update_result speed and memory
MAX_FRAME_SAVED = 300

# Load Neural Network Model for Vehicle Detection and License Plate Detection
model = YOLO(MODEL_PATH)

# Load video sample
cap = cv2.VideoCapture(INPUT_VIDEO_PATH) # Used when using video input
# cap = cv2.VideoCapture(0) # Used when using real-time camera

# Global Dictionary to Store Results
temp_results = {} # Right now is Infinity, need to limit this content
results = {} # Saved to Non-Volatile storage if reached certain number

# Global List to Store Previous Vehicle IDs
prev_vehicle_IDs = []

# Indexing variable for plate extracting purpose
plate_index = 0

# Dictionary to store tracked ID for all frame
tracked_IDs = {}
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
        # Define starting frame for video input, Run once in the program lifetime
        if frame_count < STARTING_FRAME:
            frame_count = STARTING_FRAME
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count) # HEAVY LINE, excution time: 0.09s - 0.2s

        # Setting frame Limit
        if MAX_FRAME != -1:
            if frame_count > MAX_FRAME:
                break

        # Frameskipping (HEAVY LINE, excution time: 0.09s - 0.2s)
        """
        frame_count += 1 + FRAMESKIP # i.e. at 60 fps, this advances one second
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count) # HEAVY LINE, excution time: 0.09s - 0.2s
        """

        # print Frame count
        print(f'\n======================= [Processing Frame Number: {frame_count}] =======================')

        # region Frame variables, value cleared and reused every frame

        # Saved a copy of clean frame
        clean_frame = frame.copy()

        # Create dictionary entry using frame number as key
        temp_results[frame_count] = {}

        # Variable to store detection results for vehicles
        vehicle_detecteds = []
        plate_detecteds = []

        # Create/Clear Plate number cropped Image List
        cropped_plate_list = []

        # endregion

        set_start_pin('Frame Process', show_mean=True)

        # GUARD: Detect Blurriness of the frame
        """
        set_start_pin('Blur Detection', show_mean= True)
        if not detect_image_quality(frame):
            continue
        set_end_pin('Blur Detection')
        """

        # Detect vehicles using Yolov8 COCO Trained
        set_start_pin('Model', show_mean=True)
        detections = model(frame)[0]
        set_end_pin('Model')
        
        # Loop for collecting all vehicle result in detecteds list
        for detection in detections.boxes.data.tolist():
            
            # extract bounding boxes from each result
            x1, y1, x2, y2, score, class_id = detection

            # separate bounding box for vehicles detected from all Coco object
            if int(class_id) == vehicle_class_id:

                # Add the vehicle bounding box to the list
                vehicle_detecteds.append([x1, y1, x2, y2, score])

            elif int(class_id) == plate_class_id:

                # Add the vehicle bounding box to the list
                plate_detecteds.append([x1, y1, x2, y2, score])

            print(f'Vehicle Detecteds: {vehicle_detecteds}')

        # Only continue if vehicle detected
        if vehicle_detecteds:

            # Track Vehicles using SORT
            set_start_pin('Tracker')
            vehicle_IDs = motion_tracker.update(np.asarray(vehicle_detecteds))
            set_end_pin('Tracker')

            

            # Loop for collecting all license plate result in license plates list
            for license_plate in plate_detecteds:

                # extract bounding boxes from each result
                x1, y1, x2, y2, score = license_plate

                # Skip detected plate if the confidence number is too low
                if score > 0.4:

                    # Get Car bounding box for each license plate
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, vehicle_IDs)

                    # continue the script only if car is found for the specific license plate
                    if (car_id != -1):

                        # crop license plate image 
                        license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2)]
                        # img_ori = license_plate_crop.copy()

                        # Enhancing each plate Image before combined
                        # license_plate_crop = img_enhancer.predefined_filters(license_plate_crop)

                        # read license plate number using OCR Module
                        set_start_pin('OCR', True)
                        license_plate_bboxes, license_plate_text, license_plate_text_score,  = OCRv2(license_plate_crop)
                        set_end_pin('OCR')
                        

                        # if text sucessfully Read, Add all the necessary data to final results Dictionary
                        if license_plate_text is not None:
                            
                            # region Temporary Segmentation
                            """
                            car_crop = frame[int(ycar1) : int(ycar2), int(xcar1) : int(xcar2)]

                            if car_crop.shape[0] > 30 and car_crop.shape[1] > 30:

                                car_color = color_categorizer(find_dominant_color(car_crop))

                                # save cropped license plate image to file with indexed names
                                cv2.imwrite(f'{OUTPUT_PATH}/{frame_count}_[{license_plate_text}]_[{car_color}].jpg', car_crop)

                                # increase the index by 1 for each license plate
                                plate_index += 1
                            """
                            
                            """
                            if license_plate_bboxes:
                                for bbox in license_plate_bboxes:
                                    
                                    # Guard for Rogue bbox data
                                    if bbox == (0,0):
                                        continue

                                    # Extract BBox and Car ID data from Vehicle_ID
                                    x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[2][1]
                                    
                                    # Draw Bounding boxes for each car in Detected List
                                    cv2.rectangle(license_plate_crop, (int(x1), int(y1)), (int(x2), int(y2)),  colorGreen, 3)

                            # save cropped license plate image to file with indexed names
                            cv2.imwrite(f'{OUTPUT_PATH}/{frame_count}_[{car_id}]_[{license_plate_text}]_[{license_plate_text_score}].jpg', license_plate_crop)

                            # increase the index by 1 for each license plate
                            plate_index += 1
                            """
                            # endregion

                            # Crop Cars from frame
                            car_crop = frame[int(ycar1) : int(ycar2), int(xcar1) : int(xcar2)]
                            
                            car_color = "Unknown"

                            # Get Color from cropped car image
                            if car_crop.shape[0] > 0 and car_crop.shape[1] > 0:
                                car_color = color_categorizer(find_dominant_color(car_crop))

                            # Delete earlier frame data if max frame is reached, Significantly affect update_result speed and memory
                            if len(temp_results) > MAX_FRAME_SAVED:
                                del temp_results[min(temp_results.keys())]

                            # Add data aquired to temporary result before filtered further    
                            temp_results[frame_count][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2],
                                                                         'color': car_color}, 
                                                            'license_plate': {'bbox': [x1, x2, y1, y2],
                                                                            'text': license_plate_text, 
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score}}                  

            set_start_pin('Result Update')
            tracked_id_checker(tracked_IDs, vehicle_IDs, temp_results)
            set_end_pin('Result Update')
    
            # region Concatenate
            """
            set_start_pin('Concatenate Process')

            # Concenate All plate captured in frame to reduce OCR iteration
            if cropped_plate_list:
        
                # Get max height of the image from cropped plate list
                # max_height = max(img.shape[0] for img in cropped_plate_list)

                # Get max width of the image from cropped plate list
                max_width = max(img.shape[1] for img in cropped_plate_list)

                # Filters separation tag to match concatenate requirements
                filtered_tag = img_enhancer.predefined_filters(TAG)

                # Resize separation tag to match concatenate requirements
                # resized_tag = cv2.resize(filtered_tag, (int(filtered_tag.shape[1] * max_height / filtered_tag.shape[0]), max_height), interpolation = cv2.INTER_CUBIC)
                resized_tag = cv2.resize(filtered_tag, (max_width, int(filtered_tag.shape[0] * max_width / filtered_tag.shape[1])), interpolation = cv2.INTER_CUBIC)
                
                # List to save all resized plate list. Cleared for each frame
                resized_plate_list = []

                # Resize every image in cropped size list
                i = 0
                for img in cropped_plate_list:
                    # resized_plate_list.append(cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height), interpolation = cv2.INTER_CUBIC))
                    resized_plate_list.append(cv2.resize(img, (max_width, int(img.shape[0] * max_width / img.shape[1])), interpolation = cv2.INTER_CUBIC))
                    
                    if (i != len(cropped_plate_list) - 1):
                        resized_plate_list.append(resized_tag)

                    i += 1

                # Concatenate every plate in the frame into single image Horizontally
                # combined_plate = cv2.hconcat(resized_plate_list)

                # Concatenate every plate in the frame into single image Vertically
                combined_plate = cv2.vconcat(resized_plate_list)

                # cv2.imshow('Combined', combined_plate)
                # cv2.waitKey(0)

                set_end_pin('Concatenate Process')

                # read license plate number using OCR Module
                set_start_pin('OCR', show_mean=True)
                license_plate_text, license_plate_text_score = OCRv2(combined_plate)
                set_end_pin('OCR')

                # region Temporary License Plate Segmentation

                if license_plate_text is not None:
                
                    # save cropped license plate image to file with indexed names
                    cv2.imwrite(f'../Data/playground/{plate_index}_[{license_plate_text}]_[{int(license_plate_text_score * 100)}].jpg', combined_plate)

                # increase the index by 1 for each license plate
                plate_index += 1

                # endregion
            """
            # endregion

            # region Data Visualization
            """
            set_start_pin('Visualization')
            # dupliicate the original frame
            display_frame = frame.copy()
            
            if vehicle_detecteds:
                # Draw Bounding boxes for each car in Detected List
                for detection in vehicle_IDs:

                    # Extract BBox and Car ID data from Vehicle_ID
                    x1, y1, x2, y2, car_id = detection

                    # Draw Bounding boxes for each car in Detected List
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)),  (255, 0, 0), 3)

                    # Add Car ID to the Car BBox and License plate BBox
                    cv2.putText(display_frame, f'{int(car_id)}', (max(0, int(x1)), max(40, int(y1))), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            if plate_detecteds:
                # Draw bounding boxes for each plate in License_plates List
                for license_plate in plate_detecteds:

                    # extract bounding boxes from each result
                    x1, y1, x2, y2, score = license_plate

                    # Draw Bounding boxes for each car in Detected List
                    cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)),  (0, 255, 0), 3)

                    # Add Car ID to the Car BBox and License plate BBox
                    cv2.putText(display_frame, f'{score}', (max(0, int(x1)), max(40, int(y1))), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

                    # Display License Plate ID
                    # xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, vehicle_IDs)
            
            # Show the image
            display_frame = cv2.resize(display_frame, (960, 540), interpolation = cv2.INTER_AREA)
            cv2.imshow('Detection', display_frame)
            cv2.waitKey(1)
            set_end_pin('Visualization')
            """
            
            
            
            
            # endregion

            # region Licensce Plate Segmentations
            """
            # crop each License plate from original image
            for license_plate in license_plates.boxes.data.tolist():

                # extract bounding boxes from each result
                x1, y1, x2, y2, score, class_id = license_plate

                # if license plate score too low, skip this data
                if score < 0.5:
                    continue

                # cropped plate using bounding box
                cropped_plate = clean_frame[int(y1):int(y2), int(x1):int(x2)]

                # cropped_plate = img_enhancer.sharpen(cropped_plate)

                # save cropped license plate image to file with indexed names
                cv2.imwrite(f'../Data/PlateNumbers (margonda 3)/{plate_index}.jpg', cropped_plate)

                # increase the index by 1 for each license plate
                plate_index += 1
            # """
            # endregion

        set_end_pin('Frame Process')
    
    else:
        break
# endregion

# ==================================================================================
#######################[             TRACING              ]#########################
# ==================================================================================

# region Tracing
# displaying the memory
# print(f'tracemalloc output: {tracemalloc.get_traced_memory()}')

# Take memory allocation data using tremalloc
"""
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

print("[ Top 10 ]")
for stat in top_stats[:10]:
    print(stat)
 
# stopping the library
tracemalloc.stop()
"""

# endregion


for key, value in results.items():
    print(key, ":", value)

set_end_pin('Total Process')

# write results into CVS File
# write_csv(temp_results, '../Data/csv/Test.csv')


