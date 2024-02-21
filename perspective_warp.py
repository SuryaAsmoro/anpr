import cv2
import numpy as np
import image_enhancing

# Plate number size
PLATE_WIDTH = 430
PLATE_HEIGHT = 135

# Instantialize Image Enhancing 
image_enhancer = image_enhancing.EnhancingModule()

def reorder_point(h):

    """
    Reorder point to match the order of the  warpPerspective Function

    Args:
        h (np.array(4,2)): original conftours found from FindContours Function Opencv
    """

    h = h.reshape((4,2))
    hnew = np.zeros((4,2), dtype= np.float32)
    
    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis = 1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew

def deskew(img):

    # create a copy of the image before enhancing
    img_original = img.copy()

    # Apply grayscale filter
    img = image_enhancer.grayscale(img)

    # Apply Gaussian Blur Filter
    img = image_enhancer.gaussian_blur(img)

    # Sharpen
    # img = image_enhancer.sharpen(img)

    # Apply Median Blur Filter
    # img = image_enhancer.median_blur(img, 10)

    # Apply Tresholding
    img = image_enhancer.tresholding(img)

    # Apply Auto Negative
    # img = image_enhancer.negative(img)

    # Add Paddings
    img = image_enhancer.padding(img, (0, 0, 1, 1))

    # apply Canny Edge Filter
    img_canny = image_enhancer.canny(img)

    # Find the Contour
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # ???
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find Contour who have 4 Points
    for contour in contours:

        # ???
        p = cv2.arcLength(contour, True)

        # ???
        approx = cv2.approxPolyDP(contour, 0.025*p, True)

        # if Found 4 curve that connected
        if len(approx) == 4:

            # save that specific curves as target array
            target = approx
            break

        return img_original, contour, img_canny, False
    
    # reorder point to match WarpPerspective Format
    approx = reorder_point(target)

    # prepare array with edge point position
    pts = np.float32([[0, 0], [PLATE_WIDTH, 0], [PLATE_WIDTH, PLATE_HEIGHT], [0, PLATE_HEIGHT]])

    # print(f'pts = {pts}')

    op = cv2.getPerspectiveTransform(approx, pts)
    dst = cv2.warpPerspective(img_original, op, (PLATE_WIDTH, PLATE_HEIGHT))

    return dst, target, img_canny, True
