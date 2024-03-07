import cv2
import mediapipe as mp
import numpy as np

window_width = 1366 
window_height = 750
window = np.zeros((window_height, window_width, 3), np.uint8)

paddle_width = 10
paddle_height = 100
paddle_speed = 10
player1_pos = window_height // 2 - paddle_height // 2
player2_pos = window_height // 2 - paddle_height // 2
ball_pos = [window_width // 2, window_height // 2]
ball_dir = [10, 10]
player1_score = 0
player2_score = 0


hand = mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,model_complexity=1,min_detection_confidence=0.3,min_tracking_confidence=0.5)



def findhands():
    mylmList = []
    if results_hand.multi_hand_landmarks:
        for handType, handLms in zip(results_hand.multi_handedness, results_hand.multi_hand_landmarks):
            for id, lm in enumerate(handLms.landmark):
                px, py= int(lm.x * w), int(lm.y * h)
                mylmList.append([px, py])
    return mylmList 


cap = cv2.VideoCapture(0)

while True:
    s, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results_hand = hand.process(imgRGB)
    knu = findhands()
    window.fill(0) 


    cv2.rectangle(window, (10, player1_pos), (10 + paddle_width, player1_pos + paddle_height), (255, 255, 255), -1)
    cv2.rectangle(window, (window_width - 10 - paddle_width, player2_pos), (window_width - 10, player2_pos + paddle_height), (255, 255, 255), -1)
    cv2.circle(window, (ball_pos[0], ball_pos[1]), 10, (255, 255, 255), -1)


    cv2.putText(window, f"Player 1: {player1_score}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(window, f"Player 2: {player2_score}", (window_width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

 
    ball_pos[0] += ball_dir[0]
    ball_pos[1] += ball_dir[1]

    if ball_pos[1] <= 0 or ball_pos[1] >= window_height:
        ball_dir[1] = -ball_dir[1]

    if 10 <= ball_pos[0] <= 10 + paddle_width and player1_pos <= ball_pos[1] <= player1_pos + paddle_height:
        ball_dir[0] = -ball_dir[0]

    if window_width - 10 - paddle_width <= ball_pos[0] <= window_width - 10 and player2_pos <= ball_pos[1] <= player2_pos + paddle_height:
        ball_dir[0] = -ball_dir[0]

    if ball_pos[0] <= 0:
        player2_score += 1
        ball_pos = [window_width // 2, window_height // 2]  

    if ball_pos[0] >= window_width:
        player1_score += 1
        ball_pos = [window_width // 2, window_height // 2] 

    cv2.imshow('Pong Game', window)

    key = cv2.waitKey(1)
    if len(knu) == 42:
        player1_pos = knu[20][1] + 100
        player2_pos = knu[41][1] + 100
        cv2.circle(img , (knu[41][0],knu[41][1]) , 2 , (0,0,0) , -1)
        cv2.circle(img , (knu[20][0],knu[20][1]) , 2 , (0,0,0) , -1)
    
    img = cv2.resize(img, (300,200))
    cv2.imshow("vid", img)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
