import cv2
"""
import os
# region directory correction
directory = r'c:\Temporary\Skripsi\Programs\Main Program\anpr-ui\Project Main - Plate Number Recognition'
os.chdir(directory)

# endregion
"""

class UpscalingModule:

    def slow_upscale_4x(self, img):
        sr = cv2.dnn_superres.DnnSuperResImpl.create()

        model = "../Models/Superres Model/EDSR_x4.pb"

        sr.readModel(model)

        sr.setModel("edsr", 4)

        result = sr.upsample(img)

        return result



    def slow_upscale_8x(self, img):
        sr = cv2.dnn_superres.DnnSuperResImpl.create()

        model = "../Models/Superres Model/LapSRN_x8.pb"

        sr.readModel(model)

        sr.setModel("lapsrn", 8)

        result = sr.upsample(img)

        return result

    def fast_upscale_4x(self, img):
        sr = cv2.dnn_superres.DnnSuperResImpl.create()

        model = "../Models/Superres Model/FSRCNN_x4.pb"

        sr.readModel(model)

        sr.setModel("fsrcnn", 4)

        result = sr.upsample(img)

        return result

    def fast_upscale_2x(self, img):
        sr = cv2.dnn_superres.DnnSuperResImpl.create()

        model = "../Models/Superres Model/FSRCNN_x2.pb"

        sr.readModel(model)

        sr.setModel("fsrcnn", 2)

        result = sr.upsample(img)

        return result



# region Testing
"""
from image_enhancing import *

img_upscaler = UpscalingModule()
img_enhancer = EnhancingModule()

img = cv2.imread("../Data/Image/sample.jpg")
img_ori = img.copy()

result = img_upscaler.fast_upscale_4x(img)

resized = cv2.resize(result, (960, 540), interpolation = cv2.INTER_AREA)
sharped = img_enhancer.sharpen(img, 3)

cv2.imshow("Ori", img_ori)
cv2.imshow("Image", resized)
cv2.imshow("sharpen", sharped)
cv2.waitKey(0)

"""
# endregion



