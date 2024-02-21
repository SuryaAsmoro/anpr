import cv2
import numpy as np

# CONFIG
PLATE_WIDTH = 430
PLATE_HEIGHT = 135

# Enhancing Class to Process Image before significant process.
class EnhancingModule:

    """
    Enhancing Class to Process Image before significant process.
    """

    # Apply negative filter to the image.
    def negative(self, img):

        """
        Apply negative filter to the image.

        Args:
            img (np.array): single grayscale-format or thresholded image file.

        Returns:
            img (np.array): negative image with same properties as input.
        """
        # Negate every pixel of the image using OpenCV
        neg = cv2.bitwise_not(img)

        return neg

    # Apply negative to the image if the image have High Value Mean.
    def adaptive_negative(self, img, median_treshold = 127):

        """
        Apply negative to the image if the image have High Value Mean.

        Args:
            img (np.array): single grayscale-format or thresholded image file.
            median_treshold (int): apply negative if the image's mean less than treshold

        Returns:
            img (np.array): negative image with same properties as input or The original input image.
        """
        # Find average value of the given Image
        mean = np.average(img)

        # only apply negative filter if the average value is more than median_treshold
        if median_treshold > 127:
            img = self.negative(img)

        return img

    # Convert RGB-channel image to single-channel grayscale image. 
    def grayscale(self, img):

        """
        Convert RGB-channel image to single-channel grayscale image. 

        Args:
            img (np.array): single RGB-format image file.

        Returns:
            img (np.array): grayscale image with same properties as input.
        """

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    # Apply median blur filter to input image, used to remove Noise.
    def median_blur(self, img, kernel_size = 5):

        """
        Apply median blur filter to input image, used to remove Noise.

        Args:
            img (np.array): single grayscale-format or RGB image file.
            kernel_size (int): block size of pixel in which each operation will be executed.

        Returns:
            img (np.array): image with same properties as input.
        """
        
        # Apply median blur using OpenCV library
        clean = cv2.medianBlur(img, kernel_size)

        return clean

    # Apply tresholding filter using OTSU Binarization
    def tresholding(self, img):

        """
        Best tresholding filter using OTSU Binarization, 
        adaptive to lighting but a bit slower than other thresholding methods,
        works best with Gaussian Blur filter.

        Args:
            img (np.array): single grayscale-format image file.

        Returns:
            img (np.array): binary-value image with same properties as input.
        """

        # apply threshold filter using Opencv
        tresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        return tresh
    
    # Apply thresholding filter using manual inputed threshold.
    def binary_tresholding(self, img, treshold_median = 127):

        """
        Apply thresholding filter using manual inputed threshold.

        Args:
            img (np.array): single grayscale-format image file.
            threshold_median (int): median number to define binary output of a pixel.

        Returns:
            img (np.array): binary-value image with same properties as input.
        """

        # apply threshold filter using Opencv
        tresh = cv2.threshold(img, treshold_median, 255, cv2.THRESH_BINARY_INV)

        return tresh

    # Apply thresholding filter using adaptive threshold method
    def adaptive_tresholding(self, img, block_size = 11, constant = 2):

        """
        Apply thresholding filter using adaptive threshold method.

        Args:
            img (np.array): single grayscale-format image file.
            block_size (int): block size to determine neighbour pixel before applying threshold.
            constant (int): constant subtracted from neighbour mean before applying threshold.

        Returns:
            img (np.array): binary-value image with same properties as input.
        """

        # Apply threshold filter using Adaptive Gaussian OpenCV
        tresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant)
        
        return tresh

    # Apply dilataion morphology
    def dilate(self, img, block_size = 5, iterations = 1):

        """
        Apply dilatation morphology to a Binary Image

        Args:
            img (np.array): single Binary image file.
            block_size (int): block size to determine neighbour pixel before applying dilatation.
            iterations (int): define number of iteration that dilatation applied.

        Returns:
            img (np.array): binary-value image with same properties as input.
        """
        
        # Define kernel size
        kernel = np.ones((block_size, block_size), dtype=int)

        # Dilate the image using pre-defined kernel size
        dilated = cv2.dilate(img, kernel, iterations)

        return dilated

    # Apply erosion morphology
    def erode(self, img, block_size = 5, iterations = 1):

        """
        Apply Erosion morphology to a Binary Image

        Args:
            img (np.array): single Binary image file.
            block_size (int): block size to determine neighbour pixel before applying Erosion.
            iterations (int): define number of iteration that erosion applied.

        Returns:
            img (np.array): binary-value image with same properties as input.
        """
        
        # Define kernel size
        kernel = np.ones((block_size, block_size), dtype=int)

        # Erode the image using pre-defined kernel size
        eroded = cv2.erode(img, kernel, iterations)
        return eroded

    def opening(self, img):

        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """

        kernel = np.ones((5,5), dtype=int)
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return opened

    def canny(self, img):

        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """

        outlined = cv2.Canny(img, 30, 150)
        return outlined
    
    def gaussian_blur(self, img):

        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """

        blur = cv2.GaussianBlur(img, (5,5), 0)
        return blur
    
    def sharpen(self, img, kernel_size = 3):

        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """
        # Automatically cereate kernel matrix with defined size
        kernel = np.full([kernel_size, kernel_size], -1)
        kernel[int(kernel_size//2),int(kernel_size//2)] = int(kernel_size * kernel_size)
        sharpened_image = cv2.filter2D(img, -1, kernel)

        return sharpened_image
    
    def resize(self, img, scale):

        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """

        resized_img = cv2.resize(img, None, fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
        return resized_img

    def padding(self, img, thickness):

        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """

        constant = cv2.copyMakeBorder(img, thickness[0], thickness[1], thickness[2], thickness[3], cv2.BORDER_CONSTANT, value=0)
        return constant

    def deskew(self, img):

        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """

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

        # create a copy of the image before enhancing
        img_original = img.copy()
        
        # Apply grayscale filter
        img = self.grayscale(img)

        # Apply Gaussian Blur Filter
        img = self.gaussian_blur(img)

        # Apply Tresholding
        img = self.tresholding(img)

        # Apply Auto Negative
        # img = self.negative(img)

        # Apply Median Blur Filter
        # img = self.remove_noise(img)

        # Add Paddings
        # img = self.padding(img, 10)

        # apply Canny Edge Filter
        img_canny = self.canny(img)

        # apply dilatation
        # img_canny = self.dilate(img, 5, 1)

        # Find the Contour
        contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find Contour who have 4 Points
        for contour in contours:

            # ???
            p = cv2.arcLength(contour, True)

            # ???
            approx = cv2.approxPolyDP(contour, 0.02*p, True)

            # if Found 4 curve that connected
            if len(approx) == 4:

                # save that specific curves as target array
                target = approx
                break

            return img_original, False
        
        # reorder point to match WarpPerspective Format
        approx = reorder_point(target)

        # 
        pts = np.float32([[0, 0], [PLATE_WIDTH, 0], [PLATE_WIDTH, PLATE_HEIGHT], [0, PLATE_HEIGHT]])

        # print(f'pts = {pts}')

        op = cv2.getPerspectiveTransform(approx, pts)
        dst = cv2.warpPerspective(img_original, op, (PLATE_WIDTH, PLATE_HEIGHT))

        return dst, True

    def predefined_filters(self, img):
    
        img = self.resize(img, 2)
        img = self.grayscale(img)
        # img = self.median_blur(img, 3)
        # img = self.sharpen(img, 3)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = self.tresholding(img)
        return img




