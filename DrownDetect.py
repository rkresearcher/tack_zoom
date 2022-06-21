import object_detection as cv
from object_detection import draw_bbox
import cv2
import numpy as np
def zoom_center(img, x_avg,y_avg,zoom_factor):

    y_size = img.shape[0]
    x_size = img.shape[1]
    # define new boundaries
    if x_avg >=x_size*0.5:
        zoom_factor = 1
        x_avg = 100
        x2 = x_size
    else:
        zoom_factor = zoom_factor
        x_avg = 0
        x2 = x_size-100
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))
      # define new boundaries
  
    # first crop image then scale
    img_cropped = img[y1:y2,x_avg:x2]
    
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)
    #return (img_cropped)
webcam = cv2.VideoCapture('test2.mp4')
ret, frame = webcam.read()
if not webcam.isOpened():
    print("Could not open webcam")
 #gives time in seconds after 1970

#variable dcount stands for how many seconds the person has been standing still for


#this loop happens approximately every 1 second, so if a person doesn't move,
#or moves very little for 10seconds, we can say they are drowning
z = 1
#loop through frames
while webcam.isOpened():
    # read frame from webcam
    status, frame = webcam.read()        
    bbox, label, x_avg,y_avg = cv.detect_common_objects(frame)
    
    x_min = min(bbox)
    y_min = min(bbox)
    x_max = max(bbox)
    y_max = max(bbox)
    out = draw_bbox(frame, bbox, label)
         
    out = zoom_center(out,x_avg,y_avg,z)
    cv2.imshow("Real-time object detection", out)
    z = z+0.0002
    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
webcam.release()
cv2.destroyAllWindows()
