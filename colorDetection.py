import cv2
import numpy as np
from algorithm_analyzer import *
from image_enhancing import *
from enum import Enum
import colorsys
# from sklearn.cluster import KMeans
# import sklearn
 
class HighSaturatedColor(Enum):

    # These Colors Categorized by Hue
    BLUE = (100, 124)
    CYAN = (85, 99)
    GREEN = (40, 84)
    YELLOW = (25, 39)
    ORANGE = (17, 24)
    RED_A = (170, 180)
    RED_B = (0, 16)
    PINK = (145, 169)
    PURPLE = (125, 144)

class LowSaturatedColor(Enum):

    # These 3 Color Categorized by Value
    BLACK = (0, 64)
    WHITE = (172, 255)
    SILVER = (65, 171)

# region V3

# Instantiate Enhancing module
image_enhancer = EnhancingModule()

def kmean_cluster_v01(data, number_cluster = 4):

    if len(data) > number_cluster:

        # State the criteria to stop the algorithm.
        # detail: stop clustering when epsilon accuracy == 1.0, and iteration reached 10)
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 10, 1.0)

        # determine how initial centers are taken.
        # detail: use random point to initialize first center
        flags = cv2.KMEANS_RANDOM_CENTERS
        # flags = cv2.KMEANS_PP_CENTERS

        set_start_pin('KMean Clustering', show_mean= True)
        compactness, labels, centers = cv2.kmeans(data, number_cluster, None, criteria, 10, flags)
        set_end_pin('KMean Clustering')

    return compactness, labels, centers

# Create Color image Bar
def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    # red, green, blue = int(color[2]), int(color[1]), int(color[0])

    return bar

def find_dominant_color(img):

    # Resize the image to Reduce the clustering data
    # if img.shape[0] > 400 and img.shape[1] > 100:
       # img = image_enhancer.resize(img, 0.15)

    # img_orishape = img.shape

    set_start_pin('Image Reshaping', show_mean= True)
    img = cv2.resize(img, (80, 60), interpolation = cv2.INTER_LINEAR)
    set_end_pin('Image Reshaping')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get image height and width
    height, width = np.shape(img)[0], np.shape(img)[1]

    # Do image complexity reduction Here ... or up above
    # reshape data into 3 Column array (Prepare this data to reduce calculation)
    data = np.reshape(img, (height * width, 3))

    # Change datatype into float 32 for further calculation
    data = np.float32(data)

    # Use Kmeans to extract Clusters
    compactness, labels, centers = kmean_cluster_v01(data, number_cluster=4)

    # region Dominant cluster selection
    # Prepare memory for counting cluster
    colors_dict = {}

    # variable to store max saturation
    max_saturation = (-1, (-1, -1, -1))

    # variable to store selected method
    rgb_value = (-1, -1, -1)

    for label in labels:

        if label[0] in list(colors_dict.keys()):
            colors_dict[label[0]] += 1
        else:
            colors_dict[label[0]] = 1

    for index in range(len(centers)):
        
        # Get HSV Value from rgb_value
        hsv_value = colorsys.rgb_to_hsv(int(centers[index][0])/float(255), int(centers[index][1])/float(255), int(centers[index][2])/float(255))

        if hsv_value[1] > max_saturation[1][1]:
            max_saturation = (index, hsv_value)
    
    # Get label with most pixel count
    max_value = max(colors_dict.values())
    max_label = [key for key, value in colors_dict.items() if value == max_value]

    print(f'color_value: {max_saturation[1][2]}')

    # If too dark, just skip
    # max_saturation[1][2] < 0.92 or max_saturation[1][1] > 0.5) and 
    # if not too dark, accept all
    if max_saturation[1][2] > 0.15 and max_saturation[1][1] > 0.25:
        rgb_value = centers[max_saturation[0]]
    else:
        rgb_value = centers[max_label][0]
    
    # endregion

    return rgb_value # , img_combined

def color_categorizer(rgb_value):
    
    # Convert RGB Values to int
    rgb_value = (int(rgb_value[0]), int(rgb_value[1]), int(rgb_value[2]))

    # Get HSV Value from rgb_value
    hsv_value = colorsys.rgb_to_hsv(rgb_value[0]/float(255), rgb_value[1]/float(255), rgb_value[2]/float(255))

    # Formatting HSV Value to (190, 255, 255)
    hsv_value = (int(hsv_value[0] * float(180)), int(hsv_value[1] * float(255)), int(hsv_value[2] * float(255)))
    
    print(f'rgb_value: {rgb_value}')
    print(f'hsv_value: {hsv_value}')

    # If saturation or value is too low
    if hsv_value[1] < 64 or hsv_value[2] < 64:

        # Find match in Low Saturated Color Enum
        for color in (LowSaturatedColor):

            # determine wether the car is black, white or silver using Value
            if color.value[0] <= hsv_value[2] <= color.value[1]:
                
                # If found, return color name as string
                return color.name
    
    # if saturation value is high enough
    else:
        
        # Find match in Low Saturated Color Enum
        for color in (HighSaturatedColor):

            # determine the color using Hue
            if color.value[0] <= hsv_value[0] <= color.value[1]:
                
                # If found, return color name as string
                if color.name == 'RED_A' or color.name == 'RED_B':
                    return 'RED'
                else:
                    return color.name
            
    # Raise an error if color match not found
    raise Exception('Missing Color Segment')

# endregion

