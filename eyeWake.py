#Terminal Command: 
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

#Formula for the Eye Aspect Ratio (EAR) is found in folder
def eyeAspectRatio(eye):
    #Find the vertical distances between points of interest on the eye
    A = dist.euclidean(eye[1],eye[5])
    B = dist.euclidean(eye[2],eye[4])
    
    #Find the horizontal distances between points of interest on the eye
    C = dist.euclidean(eye[0],eye[3])

    #Formula for the EAR
    ear = (A + B)/(2.0*C)

    return ear

#Formula for the Mouth Aspect Ratio (MAR) uses the same concept as the eye
def mouthAspectRatio(mouth):
    #Find the vertical distances between points of interest on the mouth
    D = dist.euclidean(mouth[1],mouth[5])
    E = dist.euclidean(mouth[2],mouth[4])

    #Find the horizontal distances between points of interest on the mouth
    F = dist.euclidean(mouth[0],mouth[3])

    #Formula for the MAR
    mar = (D + E)/(2.0*F)

    return mar

#shape-predictor is accessed from dlib for its pre-trained library of facial detection
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
args = vars(ap.parse_args())

#EAR threshold is the constant for an opened eye
ear_threshold = 0.3
#The next line counts the number of consecutive frames where the eye is closed to determine a blink
ear_consc_frames = 3
##MAR threshold is the constant for an opened mouth
mar_threshold = 0.7
#The next line counts the number of consecutive frames where the eye is closed to determine a yawn
mar_consc_frames = 17

#The counter start for number of frames, total is the number of blinks and yawn is the number of yawns
counter = 0
total = 0
yawn = 0
step = 0

#Starts the facial detector from dlib and the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#Creates indices for the left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#Creates indices for the mouth
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#Start the video stream to detect facial features
print("[INFO] starting video stream thread...")

vs = VideoStream(src = 0).start() #Uses the live webcam feature on laptop
fileStream = False
time.sleep(1.0)

#loop to record frames of the video
while True:
    #Takes the video and converts it to grayscale video for openCV
    frame = vs.read()
    frame = imutils.resize(frame, width = 630)
    frame = imutils.resize(frame, height = 400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detects faces in the gray format
    rects = detector(gray, 0)

    #Each value is evaluated for in rects
    for rect in rects:
        #Find the keypoints on the face and convert these locations to x and y coordinates to a numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #Find the co-ordinates of the eye points and calculate the EAR values for each eye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEARval = eyeAspectRatio(leftEye)
        rightEARval = eyeAspectRatio(rightEye)
        #Find the co-ordinates of the mouth points and calculate the MAR values for each eye
        mouth = shape[mStart:mEnd]
        mar = mouthAspectRatio(mouth)
        
        #Average Eye Aspect Ratio (EAR)
        ear = (leftEARval + rightEARval)/2.0

    #Recognizes the points of interest on the eye thoroughly by finding the convex hull of each eye and visualize each eye
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)

    #if statement to evaluate whethere computed EAR value is less than the threshold value, if true, counter will increase in number
        if ear < ear_threshold:
            counter += 1
    #if not, it will evaluate if the counter is greater than or equal to the number of consecutive frames for it to count as one blink
        else:
            if counter >= ear_consc_frames:
                total += 1
            counter = 0 #if not, counter will be reset again
        
    #if statement to evaluate whethere computed MAR value is less than the threshold value, if true, step will increase in number
        if mar < mar_threshold:
            step +=1
    #if not, it will evaluate if the step is greater than or equal to the number of consecutive frames for it to count as one yawn
        else:
            if step >= mar_consc_frames:
                yawn += 1
            step = 0
            
    #3 yawns is the average value to indicate drowsiness
        if yawn >= 3:
            cv2.putText(frame, "WARNING: YOU'RE GETTING DROWSY", (75, 200), cv2.FONT_HERSHEY_DUPLEX, 1, (4, 10, 150), 2)
            cv2.putText(frame, "PLEASE PULL OVER TO THE SIDE", (95, 240), cv2.FONT_HERSHEY_DUPLEX, 1, (4, 10, 150), 2)
        
    #Display the number of blinks/yawns along with the constantly calculated EAR/MAR value in the frame
        cv2.putText(frame, "Blinks: {}".format(total), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (4, 10, 150), 2)
        cv2.putText(frame, "EAR: {}".format(ear), (590, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (4, 10, 150), 2)
        cv2.putText(frame, "Yawns: {}".format(yawn), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (4, 10, 140), 2)
        cv2.putText(frame, "MAR: {}".format(mar), (590, 70), cv2.FONT_HERSHEY_DUPLEX, 0.5, (4, 10, 140), 2)
    #Frame specifics
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #Break the while loop if q(quit) is pressed
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
        
    









    

