import math
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
            num_hands=maxHands,
            running_mode=VisionRunningMode.IMAGE)
        self.landmarker = HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True, flipType=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)
        detection_result = self.landmarker.detect(mp_image)
        
        allHands = []
        h, w, c = img.shape
        if detection_result.hand_landmarks:
            for idx in range(len(detection_result.hand_landmarks)):
                handLms = detection_result.hand_landmarks[idx]
                handedness = detection_result.handedness[idx][0].category_name
                
                mylmList = []
                xList = []
                yList = []
                for lm in handLms:
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                    
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
                
                myHand = {"lmList": mylmList, "bbox": bbox, "center": (cx, cy)}
                
                if flipType:
                    myHand["type"] = "Left" if handedness == "Right" else "Right"
                else:
                    myHand["type"] = handedness
                    
                allHands.append(myHand)
                
                if draw:
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
                    for point in mylmList:
                        cv2.circle(img, (point[0], point[1]), 5, (255, 0, 255), cv2.FILLED)

        return allHands
