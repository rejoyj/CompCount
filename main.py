import cv2
import numpy as np

url = "http://192.168.252.155:8080/video"
cap = cv2.VideoCapture(url)

total_count = 0
line_x = 400

previous_centers = []

while True:

    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame = cv2.resize(frame,(800,600))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    # threshold to isolate tape
    _, thresh = cv2.threshold(blur,180,255,cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours,_ = cv2.findContours(
        thresh,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    current_centers = []

    for c in contours:

        area = cv2.contourArea(c)

        # pocket size filter
        if area < 200 or area > 1200:
            continue

        x,y,w,h = cv2.boundingRect(c)

        aspect = w/float(h)

        if 0.3 < aspect < 1.2:

            # crop slightly inside the pocket
            roi = gray[y+3:y+h-3, x+3:x+w-3]

            if roi.size == 0:
                continue

            mean_intensity = np.mean(roi)

            # if component present (darker)
            if mean_intensity < 140:

                cx = int(x+w/2)
                cy = int(y+h/2)

                current_centers.append((cx,cy))

                # draw green rectangle
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

                for px,py in previous_centers:

                    if abs(cx-px)+abs(cy-py) < 30:

                        if px < line_x and cx >= line_x:
                            total_count += 1

            else:
                # empty pocket (draw red)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    previous_centers = current_centers

    cv2.line(frame,(line_x,0),(line_x,600),(255,0,255),2)

    cv2.putText(frame,
                "Total Count: "+str(total_count),
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,0,255),2)

    cv2.imshow("Detector",frame)
    cv2.imshow("Threshold",thresh)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()