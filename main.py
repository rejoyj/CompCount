import cv2
import numpy as np

url = "http://192.168.252.155:8080/video"
cap = cv2.VideoCapture(url)

total_count = 0
line_position = 300

previous_centers = []

while True:

    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame = cv2.resize(frame,(800,600))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray,(7,7),0)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        blur,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,2)

    # Morphological cleanup
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    current_centers = []

    for c in contours:

        area = cv2.contourArea(c)

        # Filter noise
        if area < 80 or area > 5000:
            continue

        # Detect shape
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # Only rectangles (4 corners)
        if len(approx) == 4:

            x,y,w,h = cv2.boundingRect(approx)
            aspect = w / float(h)

            if 0.5 < aspect < 3:

                cx = int(x + w/2)
                cy = int(y + h/2)

                current_centers.append((cx,cy))

                # Draw rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

                # Draw center
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

                # Compare with previous frame
                for px,py in previous_centers:

                    distance = abs(cx-px) + abs(cy-py)

                    if distance < 20:

                        if py < line_position and cy >= line_position:
                            total_count += 1

    previous_centers = current_centers

    # Draw counting line
    cv2.line(frame,(0,line_position),(800,line_position),(255,0,255),2)

    # Display count
    cv2.putText(frame,
                "Total Count: "+str(total_count),
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,0,255),2)

    cv2.imshow("SMD Detector",frame)
    cv2.imshow("Threshold",thresh)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()