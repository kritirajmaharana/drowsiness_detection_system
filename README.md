# python drowsiness_detection_system_by_kritiraj
#### The code implements a drowsiness detection system using facial landmarks and triggers an alarm ifthe person is detected to be drowsy or yawning.The message that the alarm function will speak out loud when triggered
    
## Dependencies
1. python 3
2. opencv
3. dlib
4. imutils
5. scipy
6. argparse
7. numpy

#### 01. The code is importing necessary libraries and modules for the drowsiness detection system.


     from scipy.spatial import distance as dist
     from imutils.video import VideoStream
     from imutils import face_utils
     from threading import Thread
     import numpy as np
     import argparse
     import imutils
     import time
     import dlib
     import cv2
     import os



 #### 02. The function "alarm" sets off an alarm by using the "espeak" command to speak a given message whilea global variable "alarm_status" is true, and sets another global variable "saying" to true while adifferent global variable "alarm_status2" is true    
 #### :param msg: The message that the alarm function will speak out loud
     def alarm(msg):
         global alarm_status
         global alarm_status2
         global saying

         while alarm_status:
            print('call')
            s = 'espeak "' + msg + '"'
            os.system(s)

        if alarm_status2:
            print('call')
            saying = True
            s = 'espeak "' + msg + '"'
            os.system(s)
            saying = False
 
#### 03 The function calculates the eye aspect ratio and returns the average ratio along with the left andright eye coordinates.
    
#### :param eye: The parameter "eye" is a list of 6 tuples representing the coordinates of the landmarksof an eye. The landmarks are ordered as follows: the leftmost point of the eye (index 0), followed by the points on the top eyelid (indexes 1-2-3), and
#### :return: The function `final_ear` returns a tuple containing the eye aspect ratio (ear) and the coordinates of the left and right eyes.
     def eye_aspect_ratio(eye):
         A = dist.euclidean(eye[1], eye[5])
         B = dist.euclidean(eye[2], eye[4])

         C = dist.euclidean(eye[0], eye[3])

         ear = (A + B) / (2.0 * C)

         return ear
         
#### 04 The function calculates the eye aspect ratio and returns it along with the left and right eye landmarks.
#### :param shape: The "shape" parameter is a NumPy array containing the facial landmark coordinatesdetected by a facial landmark detector. These coordinates correspond to specific points on the face,such as the corners of the eyes, nose, and mouth. The function uses the coordinates of the left andright eyes to calculate the eye
#### :return: The function `final_ear` returns a tuple containing the average eye aspect ratio (`ear`)and the coordinates of the left and right eyes (`leftEye` and `rightEye`).         
    def final_ear(shape):
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
 
        ear = (leftEAR + rightEAR) / 2.0
        return (ear, leftEye, rightEye)
        
#### 05. The function calculates the distance between the upper and lower lips based on the coordinates offacial landmarks.   
#### :param shape: The "shape" parameter is likely a numpy array containing the facial landmarks detectedby a facial landmark detection algorithm. The landmarks are typically represented as (x, y)coordinates on the face. The function "lip_distance" uses these landmarks to calculate the distancebetween the top and bottom lips
#### :return: the distance between the upper and lower lips based on the input shape.
    def lip_distance(shape):
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))

        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))

        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)

        distance = abs(top_mean[1] - low_mean[1])
        return distance
#### 06.   The code block initializes the argument parser `ap` to accept command line arguments. Specifically,' it allows the user to specify the index of the webcam to use for the video stream. The default value' is 0, which corresponds to the default webcam on the system.      
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0,   //For external webcam, use the webcam number accordingly eg. (1) (2) like this
                help="index of webcam on system")
    args = vars(ap.parse_args())

    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 30
    YAWN_THRESH = 30
    alarm_status = False
    alarm_status2 = False
    saying = False
    COUNTER = 0

    print("-> Loading the predictor and detector...")
    detector = dlib.get_frontal_face_detector()
    # detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  # cv2 face detector Faster but less accurate I am using dlib one use can toggle between them
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # dlib face detector slower but accurate

    print("-> Starting Video Stream")
    vs = VideoStream(src=args["webcam"]).start()
    # vs= VideoStream(usePiCamera=True).start()     # For Raspberry Pi
    time.sleep(1.0)
    
#### 07. This code block is the main loop of the program. It continuously reads frames from the video stream,resizes the frame, converts it to grayscale, and detects faces in the frame using the dlib facedetector. For each detected face, the program calculates the eye aspect ratio and lip distance todetermine if the person is drowsy or yawning. If the person is drowsy or yawning, an alarm is triggered using the `alarm` function. The program also displays the eye aspect ratio and lip distance on the frame, along with any alerts. The loop continues until the user presses the "q" keyto quit the program.    
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=650)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)  # if you using dlib face detector then use this line of code
        # rects = detector.detectMultiScale(gray, scaleFactor=1.1,   # if you using cv2 face detection then use these code and uncomment it
        #                                   minNeighbors=5, minSize=(30, 30),
        #                                   flags=cv2.CASCADE_SCALE_IMAGE)

        for rect in rects:  # this line of code is for dlib face detector
        # for (x, y, w, h) in rects:  # these below two lines are for cv2 face detector
        #     rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            eye = final_ear(shape)
            ear = eye[0]
            leftEye = eye[1]
            rightEye = eye[2]

            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
 
            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if alarm_status == False:
                        alarm_status = True
                        t = Thread(target=alarm, args=('wake up sir',))
                        t.deamon = True
                        t.start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                COUNTER = 0
                alarm_status = False

            if (distance > YAWN_THRESH):
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if alarm_status2 == False and saying == False:
                    alarm_status2 = True
                    t = Thread(target=alarm, args=('take some fresh air sir',))
                    t.deamon = True
                    t.start()
            else:
                alarm_status2 = False
 
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()
    
Change the threshold values according to your need , you can see these lines of code in above document no 06.
```
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 30`	// change this according to the distance from the camera
```    
## Author
**Kritiraj Maharana** 
