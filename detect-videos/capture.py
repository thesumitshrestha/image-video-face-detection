from datetime import date, datetime
from os import stat
import cv2
import time
import pandas as pd

# We need 4 frames for this

first_frame = None
status_list = [None, None]
times = []
df = pd.DataFrame(columns=["Start", "End"])
# Opening the camera
video = cv2.VideoCapture(0)


# Looping until q is pressed
while True:
    # Reading the video
    check, frame = video.read()

    # No Motion Detection
    status = 0

    # Changing to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Making blurry
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Assigning first frame to gray
    if first_frame is None:
        first_frame = gray
        continue

    # Second frame as delta
    delta_frame = cv2.absdiff(first_frame, gray)

    # Third frame for Threshold
    thresh_frame = cv2.threshold(delta_frame, 70, 255, cv2.THRESH_BINARY)[1]

    # Smoothing thresh_data
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Creating contours and storing it on tuples for the white part in thresh_data
    (cnts, _) = cv2.findContours(thresh_frame.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 500:
            continue

        status = 1

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    status_list.append(status)
    times.append(datetime.now())
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

        # Displaying the video
    cv2.imshow("GrayFrame", gray)
    cv2.imshow("DeltaFrame", delta_frame)
    cv2.imshow("ThreshData", thresh_frame)
    cv2.imshow("Color Frame", frame)
    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

print(status)

for i in range(0, len(times), 2):
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("Times.cvs")
video.release()
cv2.destroyAllWindows()
