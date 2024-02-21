# import pytesseract
# import imageEnhancing
import easyocr
import os
from string_formating import *
from algorithm_analyzer import *
# from paddleocr import PaddleOCR

# directory = r'c:\Temporary\Skripsi\Programs\Main Program\anpr-ui\Project Main V2'
# os.chdir(directory)

# Assign the reader module as EasyOCR Object
reader = easyocr.Reader(['en'])

# Assign reader module as PaddleOCR Object
# paddleReader = PaddleOCR(lang='en') # need to run only once to download and load model into memory

# Add pyTesseract Path from local devices
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# execution time: 0.09s
def OCRv0(img):
    """
    Read text presented in the image. Return text as output

    Args:
        img (np.array): an image containing text. Yield better result if proecssed before hand
    
    # region PyTesseract
    
    # Read the Image using OCR Model with PSM 6
    custom_config = r'--oem 3 --psm 8'
    result = pytesseract.image_to_string(img, config=custom_config)
    # endregion

    return result, 1
    
    """

def OCRv1(img):
    """
    Read text presented in the image. Return text as output

    Args:
        img (np.array): an image containing text. Yield better result if proecssed before hand
    """

    # region Image Enhancing
    # Instantiate required modules
    # conditioningModule = imageConditioning.ConditioningModule()

    # Conditioning the Image before get into OCR Model
    # img = conditioningModule.grayscale(img)
    # img = conditioningModule.tresholding(img)
    # img = conditioningModule.negative(img)
    # endregion

    # region PyTesseract
    """
    # Read the Image using OCR Model with PSM 6
    custom_config = r'--oem 3 --psm 8'
    result = pytesseract.image_to_string(conditionedImg, config=custom_config)
    """
    # endregion

    # region Easy OCR
    # Perform OCR on the preprocessed image
    resultText = reader.readtext(img, batch_size=10)

    # Extract and concatenate the detected texts
    text = ' '.join(result[1] for result in resultText)

    text = remove_nonalphanumeric(text)
    text = text.upper()
    # endregion

    print(f'Result Text: {text}')

    return text, 1

# using easyOCR
def OCRv2(img):
    detections = reader.readtext(img, batch_size=10, width_ths=0.85)

    # Perfect Detection, the output is 2 or less
    if detections:
        for detection in detections:

            # Unpack data from detections result
            bbox, text, score = detection

            result = ocr_result_assembler(bbox, text, score)

            if result[0]:

                bbox, text, score = result[1], result[2], result[3]

                return bbox, text, score

            else:
                continue

    return ((0, 0),(0, 0), (0, 0), (0, 0)), None, 0 
"""
# using paddleOCR
def OCRv3(img):
    
    # Perform OCR on the preprocessed image
    result = paddleReader.ocr(img, cls=False)

    for detection in result:
            
        # Unpack data from detections result
        bbox, text, score = detection[0], detection[1][0], detection[1][1]

        # Compile result to Plate Format
        result = ocr_result_assembler(bbox, text, score)

        # If reault compiled successfully
        if result[0]:
            
            # Save unpacked data
            bbox, text, score = result[1], result[2], result[3]

            return bbox, text, score

        else:
            continue

    return ((0, 0),(0, 0), (0, 0), (0, 0)), None, 0 

    
    return text, score, bbox
    
def detect_bbox(img):

    bbox = (0,0,0,0)

    result = paddleReader.ocr(img,rec=False)
    result = result[0]

    return bbox
"""




def ocr_result_assembler(bbox, text, score):
            
     # GUARD for Rogue bbox data
    if bbox != (0,0):

        # Unpack bounding box from result
        x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[1][0], bbox[2][1]

        # get width/height ratio of the bounding box
        bboxRatio = (x2-x1) / (y2-y1)

        # Ideal Result if Bbox ratio Higher than 3.8
        if bboxRatio >= 3.8:
            
            # GUARD to Make sure the string is in the right format
            if license_compile_format_ideal(text):

                # remove non alphanumeric from text except hypen (-)
                text = text.upper()

                # Format the result to standard Indonesian License
                text, extra_score = format_license_ideal(text)

                # Add extra score and round up the score to integer from float
                score = int(score*100) + extra_score
                
                print(f'Result Text: {text}, Score: {score}')
                
                return True, bbox, text, score

        # Unideal result, but can be Identified as non-Civil
        elif 2.4 < bboxRatio < 3.8 :

            # GUARD to make sure the string is in the right format
            if license_compile_format_noncivil(text):
                
                # remove non alphanumeric from text except hypen (-)
                text = text.upper()

                # Format the result to standard Indonesian License
                text, extra_score = format_license_noncivil(text)

                score = int(score*100) + extra_score
                
                print(f'Result Text: {text}, Score: {score}')
                
                return True,bbox, text, score

            elif license_compile_format_special(text):

                # remove non alphanumeric from text except hypen (-)
                text = text.upper()

                # Format the result to standard Indonesian License
                text, extra_score = format_license_special(text)

                score = int(score*100) + extra_score
                
                print(f'Result Text: {text}, Score: {score}')
                
                return True, bbox, text, score
    
    return False, ((0, 0),(0, 0), (0, 0), (0, 0)), None, 0 





