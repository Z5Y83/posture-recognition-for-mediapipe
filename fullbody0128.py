import cv2
import mediapipe as mp
import numpy as np
import random as rd

def cal_angle(a,b,c):
    a=np.array(a)
    b=np.array(b)
    c=np.array(c)

    radians=np.arctan2(c[1]-b[1],c[0]-b[0])-np.arctan2(a[1]-b[1],a[0]-b[0])
    angle=np.abs(radians*180.0/np.pi)

    if angle>180.0:
        angle=360-angle
    
    return angle

def cal_distance(a,b):
    a=np.array(a)
    b=np.array(b)
    
    return np.sqrt(np.sum((a-b)**2))

mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
stage=None
stage1=None
stage2=None
r=rd.randint(0,255)
g=rd.randint(0,255)
b=rd.randint(0,255)

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # current_time=time.time()
        counter=0
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img,(520,300))               # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(img2)                  # 取得姿勢偵測結果
        canvas=np.zeros(img2.shape,dtype='uint8')    #產生新畫布
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            #取得座標
            left_eye = [landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y]
            left_hip= [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y]
            left_thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            left_fake_d = [left_heel[0]-50,left_heel[1]]

            right_eye = [landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_heel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y] 
            right_fake_d = [right_heel[0]-50,right_heel[1]]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            
            #計算有沒有跌倒的距離
            #left side
            left_distance=cal_distance(left_eye,left_heel)
            left_floor=[(abs(left_fake_d[0]-left_heel[0])),left_fake_d[1]]
            #right side
            right_distance=cal_distance(right_eye,right_heel)
            right_floor=[(abs(right_fake_d[0]-right_heel[0])),right_fake_d[1]]

            #計算有沒有跌倒的角度
            left_angle=cal_angle(left_eye,left_heel,left_fake_d)
            right_angle=cal_angle(right_eye,right_heel,right_fake_d)
            #計算蹲坐站走的角度
            angle3=cal_angle(left_hip,left_knee,left_ankle)
            angle4=cal_angle(right_hip,right_knee,right_ankle)
            
            #計算蹲坐站走的距離
            distance1=cal_distance(right_hip,right_index)
            distance2=cal_distance(left_hip,left_index)
            distance3=cal_distance(left_knee,left_index)
            distance4=cal_distance(right_knee,right_index)

            #判斷蹲坐站走和跌倒
            if left_angle<=60 or right_angle<=60:
                stage='fallen'
            else:
                if angle3>170 and angle4>170:
                    stage="stand"
                elif angle3>170 and angle4<170 or angle3<170 and angle4>170:
                    stage="walk"
                else:
                    if distance3>distance2 and distance4>distance1:
                        stage="sit"
                    else:
                        stage="squat"
                #stage='safe'
            
            #判斷有沒有舉手
            if (left_wrist[1]<left_eye[1]) or (right_wrist[1]<right_eye[1]):
                if left_wrist[1]<left_eye[1]:
                    counter+=1
                if right_wrist[1]<right_eye[1]:
                    counter+=1
                if counter==1:
                    stage1="Raise one hand"
                elif counter==2:
                    stage1="Raise two hand"
            else:
                stage1=" "
            
            #判斷有沒有揮手
            distance5=cal_distance(previous_left_wrist,left_wrist)*1000
            distance6=cal_distance(previous_right_wrist,right_wrist)*1000
            # distance7=cal_distance(left_eye,left_thumb)*1000
            # distance8=cal_distance(right_eye,right_thumb)*1000
            # print(previous_left_thumb,'\t',left_thumb)
            print(distance5,'\t',distance6)
            if 5<=(distance5 or distance6)<=30:
                stage2="wave hand"
            else:
                stage2=" "
 
        except:
            pass
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        cv2.putText(canvas,stage,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(r,g,b),2,cv2.LINE_AA)
        cv2.putText(canvas,stage1,(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(r,g,b),2,cv2.LINE_AA)
        cv2.putText(canvas,stage2,(10,70),cv2.FONT_HERSHEY_SIMPLEX,1,(r,g,b),2,cv2.LINE_AA)
        output=cv2.addWeighted(img,1,canvas,1,0)
        cv2.imshow('output', output)
        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
        
        #儲存之前的座標
        try:
            # previous_left_thumb=left_thumb
            # previous_right_thumb=right_thumb
            previous_left_wrist=left_wrist
            previous_right_wrist=right_wrist
        except:
            pass
cap.release()
cv2.destroyAllWindows()
