import cv2
import numpy as np

color_ranges = {
    'Red': [(0, 100, 100), (10, 255, 255), (0, 0, 255)],     
    'Red2': [(160, 100, 100), (180, 255, 255), (0, 0, 255)], 
    'Blue': [(100, 100, 100), (140, 255, 255), (255, 0, 0)],
    'Green': [(35, 100, 100), (85, 255, 255), (0, 255, 0)],
    'Yellow': [(15, 100, 100), (35, 255, 255), (0, 255, 255)],
    'Orange': [(5, 100, 100), (20, 255, 255), (0, 165, 255)],
    'Purple': [(130, 100, 100), (160, 255, 255), (255, 0, 255)]
}

cap = cv2.VideoCapture(0)
cap.set(3, 640)  
cap.set(4, 480)  

def detect_color(frame):
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    

    result = frame.copy()
    
    
    for color_name, (lower, upper, bgr_color) in color_ranges.items():
        if color_name == 'Red2':  
            continue
            
        
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        
    
        if color_name == 'Red':
            lower2 = np.array(color_ranges['Red2'][0])
            upper2 = np.array(color_ranges['Red2'][1])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask, mask2)
        
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        
        for contour in contours:
            
            if cv2.contourArea(contour) > 500:
                
                x, y, w, h = cv2.boundingRect(contour)
                
                
                cv2.rectangle(result, (x, y), (x + w, y + h), bgr_color, 2)
                cv2.putText(result, color_name, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, bgr_color, 2)
    
    return result

while True:
    
    success, frame = cap.read()
    if not success:
        break
    
    
    result = detect_color(frame)
    
    
    cv2.imshow('Color Detection', result)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
