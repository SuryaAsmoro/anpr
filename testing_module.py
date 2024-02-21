import cv2
import glob
import numpy as np

# ===========================================================

# region Tested Module
import image_enhancing
from ultralytics import YOLO
from perspective_warp import *
from string_formating import *
from colorDetection import *
from algorithm_analyzer import *
# endregion

# ===========================================================

# region Single Image Testing
"""
# Path to Input Image
IMAGE_PATH = 'C:/Temporary/Skripsi/Programs/Main Program//anpr-ui/Data/image/sample.jpg'

# Add input image to temporary variable
img = cv2.imread(IMAGE_PATH)

# Instantiate necessary class
enhance = image_enhancing.EnhancingModule()

# region PUT YOUR CODE HERE

license_plate_detector = YOLO(r'../Models/Yolo-Weights\surya\best.pt')

# detect license plates using License Plate Detection Model
license_plates = license_plate_detector(img)[0]

# Draw bounding boxes for each plate in License_plates List
for license_plate in license_plates.boxes.data.tolist():

    # extract bounding boxes from each result
    x1, y1, x2, y2, score, class_id = license_plate

    # Draw Bounding boxes for each car in Detected List
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),  (255, 0, 0), 3)

# endregion

img = enhance.resize(img, 0.5)
cv2.imshow("img", img)
cv2.waitKey(0)

"""

# endregion

# ===========================================================

# region Multiple Image Testing
"""
# Constant Variable
IN_FOLDER_PATH = 'C:/Temporary/Skripsi/Programs/Main Program/anpr-ui/Data/PlateNumbers (margonda 3)'
OUT_FOLDER_PATH = 'C:/Temporary/Skripsi/Programs/Main Program/anpr-ui/Data/PlateNumbers (margonda 3)'
IN_IMG_TYPE = '.jpg'
OUT_IMG_TYPE = '.jpg'

# PUT YOUR GLOBAL VARIABLE HERE
index = int(0) # Create Global variable index for naming file
image_list = [] # Create Temporary Memory to Store Images
image_enhancer = image_enhancing.EnhancingModule() # Initialize Module
successFactor = 000

# import all .jpg images from folder
for filename in glob.glob(f'{IN_FOLDER_PATH}/*{IN_IMG_TYPE}'):

    # read image and store in img
    img = cv2.imread(filename)

    # region PUT YOUR CODE HERE

    print(f'Warping Image process {index}')
    # Deskew the image
    result, contour, canny, success = deskew(img)

    if success:
        successFactor += 1

    # save canny image in folder
    cv2.imwrite(f'{OUT_FOLDER_PATH}/canny/{index}{OUT_IMG_TYPE}', canny)

    # save contoured image in folder
    contourImg = cv2.drawContours(img.copy(), contour, -1, (0, 255, 0), 3)
    cv2.imwrite(f'{OUT_FOLDER_PATH}/contour/{index}{OUT_IMG_TYPE}', contourImg)

    # save deskewed image in folder
    cv2.imwrite(f'{OUT_FOLDER_PATH}/deskewed/{index}{OUT_IMG_TYPE}', result)

    img_sharp = image_enhancer.sharpen(result, 3)
    cv2.imwrite(f'{OUT_FOLDER_PATH}/sharp/{index}{OUT_IMG_TYPE}', img_sharp)

    img_gray = image_enhancer.grayscale(result)
    img_threshold = image_enhancer.tresholding(img_gray)
    cv2.imwrite(f'{OUT_FOLDER_PATH}/thresh/{index}{OUT_IMG_TYPE}', img_threshold)
    # endregion

    # increment the index for each img
    index += 1

print(f'success rate : {successFactor/index}')
"""
# endregion

# ===========================================================

# region Video Testing
"""
# Paths
VIDEO_PATH = "C:/Temporary/Skripsi/Programs/Main Program/anpr-ui/Data/Videos/margonda/data2.mp4"
OUT_FOLDER_PATH = "C:/Temporary/Skripsi/Programs/Main Program/anpr-ui/Data/Testing/Output"

# frameskip in second
FRAMESKIP = 0

# Indexing Number for Naming File
index = int(000)

# Prepare memory for video frame capture
cap = cv2.VideoCapture(VIDEO_PATH)

# Capture frames from video with 60 frameskip
count = -1
while cap.isOpened():
    ret, img = cap.read()

    if ret:
        if FRAMESKIP > 0:
            count += 60 * FRAMESKIP # i.e. at 60 fps, this advances one second
            cap.set(cv2.CAP_PROP_POS_FRAMES, count)
        
        else:
            count += 1

        # region WRITE YOUR CODE HERE   








        # endregion

        # Write Labeled Img file to evaluation Directory
        cv2.imwrite(f'{OUT_FOLDER_PATH}/{index}.jpg', img)

        # Increment the Index
        index += 1

    else:
        cap.release()
        break

"""
# endregion

# ===========================================================

# Constant Variable
IN_FOLDER_PATH = 'C:/Temporary/Skripsi/Programs/Main Program/anpr-ui/Data/Cars'
OUT_FOLDER_PATH = 'C:/Temporary/Skripsi/Programs/Main Program/anpr-ui/Data/playground'
IN_IMG_TYPE = '.jpg'
OUT_IMG_TYPE = '.jpg'

# PUT YOUR GLOBAL VARIABLE HERE
index = int(0) # Create Global variable index for namin g file
image_list = [] # Create Temporary Memory to Store Images
image_enhancer = image_enhancing.EnhancingModule() # Initialize Module
successFactor = 000

# import all .jpg images from folder
for filename in glob.glob(f'{IN_FOLDER_PATH}/*{IN_IMG_TYPE}'):

    # Print debugging Divider
    print(f'\n======================= [Processing Image Number: {index}] =======================')

    # read image and store in img
    car_crop = cv2.imread(filename)

    # region PUT YOUR CODE HERE

    if car_crop.shape[0] > 30 and car_crop.shape[1] > 30:
        set_start_pin('Color Module', show_mean= True)
        car_color, img_combined = find_dominant_color(car_crop)
        set_end_pin('Color Module')
        color_string = color_categorizer(car_color)

        # save cropped license plate image to file with index  ed names
        cv2.imwrite(f'{OUT_FOLDER_PATH}/[{index}]_[{color_string}].jpg', img_combined)

        # increment the index for each img
        index += 1

    # endregion

