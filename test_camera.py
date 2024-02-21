import cv2

# Load video sample
cap = cv2.VideoCapture(0) # Used when using video input

frame_count = -1
ret = True

# Capture frames from video with 60 frameskip
while ret:
    
    frame_count += 1

    ret, frame = cap.read()

    if ret:

        print(f'Loaded frame: {frame_count}')
        
    else:
        break
    
# releasing the video capture
cap.release()