import cv2
import mediapipe as mp
import time


win_name = 'Face Detection'
cv2.namedWindow(win_name,cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)

mpFace = mp.solutions.face_detection
face = mpFace.FaceDetection(min_detection_confidence = 0.75)
mpDraw = mp.solutions.drawing_utils

pTime = 0


while True:
    has_frame, frame = cap.read()
    if not has_frame:
        break

    imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = face.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):

            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih),
            cv2.rectangle(frame,bbox,(0,255,0),2)
            cv2.putText(frame,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]),cv2.FONT_HERSHEY_COMPLEX,2,(0,100,255),3)

            

    cTime = time.time()
    if cTime != pTime:
        fps = 1 / (cTime- pTime)
    else:
        fps = 0
    pTime = cTime

    # frame = cv2.flip(frame,1)
    cv2.putText(frame,str(int(fps)),(30,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),3)
    cv2.imshow(win_name,frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyWindow(win_name)