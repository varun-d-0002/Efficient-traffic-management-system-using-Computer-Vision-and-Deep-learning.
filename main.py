import cv2
import numpy as np
from time import sleep
import sys

largura_min=80
altura_min=80 

offset=16 

pos_line=260 

delay= 60 

detec = []
vehicles= 0

	
def center_point(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

video = cv2.VideoCapture('Traffic_lanes.mp4')
subtracao = cv2.bgsegm.createBackgroundSubtractorMOG()
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

while True:
    ret , frame1 = video.read()
    roi1 = frame1[300:1000,570:830]
    #roi2 = frame1[300:1000,845:1090]
    #roi = [roi1,roi2]
    #tempo = float(1/delay)
    #sleep(tempo)
    gray = cv2.cvtColor(roi1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),5)
    img_sub = subtracao.apply(blur)
    dilate = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx (dilate, cv2. MORPH_CLOSE , kernel)
    dilated = cv2.morphologyEx (dilated, cv2. MORPH_CLOSE , kernel)
    contour,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(roi1, (10, pos_line), (280, pos_line), (255,127,0), 3)
    for(i,c) in enumerate(contour):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(roi1,(x,y),(x+w,y+h),(0,255,0),2)
        centro = center_point(x, y, w, h)
        detec.append(centro)
        cv2.circle(roi1, centro, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(pos_line+offset) and y>(pos_line-offset):
                vehicles+=1
                cv2.line(roi1, (10, pos_line), (280, pos_line), (255,127,0), 3)
                detec.remove((x,y))
                sys.stdout = open("output.txt","w")
                print("Number of vehicles detected : "+str(vehicles))




    cv2.putText(roi1, "VEHICLE COUNT : "+str(vehicles), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),2)
    #cv2.imshow("ROI" , roi1)
    cv2.imshow("Original Video",frame1)
    result.write(frame1)


    if cv2.waitKey(1) == 27:
        break

sys.stdout.close()
video.release()
cv2.destroyAllWindows()






