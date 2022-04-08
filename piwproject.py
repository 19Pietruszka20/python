import numpy as np
from time import sleep
import cv2


blad = 6

pozycja_lini = 550

delay = 60

detec = []
licznik_samochodow = 0

kernel=None


def centrum_konturu(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture("video_Trim.mp4")
model_tla = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

while True:
    ret, frame = cap.read()
    tempo = float(1 / delay)
    sleep(tempo)

    if not ret:
        break

    fgmask=model_tla.apply(frame)
    _,fgmask=cv2.threshold(fgmask,250,255,cv2.THRESH_BINARY)

    fgmask=cv2.erode(fgmask,kernel,iterations=1)
    fgmask=cv2.dilate(fgmask,kernel,iterations=2)

    kontur,_=cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frameCopy=frame.copy()
    cv2.line(frameCopy, (25, pozycja_lini), (2500, pozycja_lini), (255, 127, 0), 2)

    for cnt in kontur:
        if cv2.contourArea(cnt)>400:
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(frameCopy,(x,y),(x+w, y+h),(0,0,255),2)
            centrum = centrum_konturu(x, y, w, h)
            detec.append(centrum)
            cv2.circle(frameCopy, centrum, 4, (0, 0, 255), -1)
            cv2.putText(frameCopy, 'Car Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0),1,cv2.LINE_AA)


    #foregroundPart=cv2.bitwise_and(frame, frame, mask=fgmask)
    for (x, y) in detec:
        if y < (pozycja_lini + blad) and y > (pozycja_lini - blad):
            licznik_samochodow += 1
            cv2.line(frameCopy, (25, pozycja_lini), (2500, pozycja_lini), (0, 127, 255), 2)
            detec.remove((x, y))
            print("car is detected : " + str(licznik_samochodow))

    #stacked=np.hstack((frame,foregroundPart,frameCopy))

    #cv2.imshow('Original Frame, Extracted foreground and Detected Cars', cv2.resize(stacked, None, fx=0.5, fy=0.5))

    cv2.imshow("Oryginalnyobraz", frame)
    cv2.imshow("Binarnyobraz",fgmask)
    cv2.imshow("Obrazdetekcji",frameCopy)
    k=cv2.waitKey(1) & 0xff
    if k==ord('q'):
        break



cv2.destroyAllWindows()
cap.release()
