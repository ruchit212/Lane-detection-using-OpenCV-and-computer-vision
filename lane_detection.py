import cv2
import numpy as np
import matplotlib.pyplot as plt



#sources  
cap = cv2.VideoCapture("F:\\lane_detection\\src\\test_video.mp4")

#functions

# find the area which we need with polygons and feed it
def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)
    masked_img = cv2.bitwise_and(image, mask)
    return masked_img



#get the lane-lines which we are finding and feed it with colors
def draw_lines(image, lines):
    image = np.copy(image)
    blank_image = np.zeros_like(image, dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness=5)

    image = cv2.addWeighted(image, 0.8, blank_image, 1, 0.0)
    return image


def process(img):
    #convert image to gray scale 
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #blur the image- remove the noise
    median_blur_image = cv2.medianBlur(gray_img, 5)
    med_val = np.median(gray_img)
    
    #get the upper & lower value of threshold    
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))

    #find the edges of the image
    edges = cv2.Canny(median_blur_image, threshold1=lower, threshold2=upper)

    height = img.shape[0]
    width = img.shape[1]
    vertices_of_ROI = np.array([(230, height), (565, 255), (1200, height)])

    masked_edges = region_of_interest(edges, vertices_of_ROI)
    
    # plt.imshow(masked_edges)
    # plt.show() 
    
      
    #houghtransform

    lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=160, lines=np.array([]), minLineLength=50, maxLineGap=30)

    new_image = draw_lines(img, lines)
    return new_image



#get the full video and find lanes
while True:
    ret, frame = cap.read()

    #check the images
    if frame is None:
        print("Error: Unable to read frame.")
        break

    frame = process(frame)
        
    cv2.imshow('lane_detection', frame)
    
    
    #close the loop    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#release the capture video
cap.release()
cv2.destroyAllWindows()
