import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


#Capturando 1 frame para detecção
cap = cv2.VideoCapture(0)
x = 0
while cap.isOpened():
    res, frame = cap.read()
    nomeDoFrame = 'frame-%d.jpg' % x
    cv2.imwrite(nomeDoFrame, frame)
    cap.release()



#Carregando imagem
imagem = cv2.imread('frame-0.jpg')

#Detector de faces
classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Convertendo para escala de cinza
imagemcinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

#Recebe o classificador e faz a função detect
deteccoes = classificador.detectMultiScale(imagemcinza,
                                           scaleFactor=1.5,
                                           minNeighbors=5,
                                           minSize=(30, 30),
                                           maxSize=(200,200))

#Devolve a posição de cada face encontrada
print(deteccoes)
print(len(deteccoes))

if len(deteccoes) != 0:
    for (x ,y, l, a) in deteccoes:
        x = x
        y = y
        w = l
        h = a
else:
    print('Não achei uma face!')

track_window = (x, y, w, h)

#Conversao e criação do histograma
roi = imagem[y: y+h, x: x+w]
roi = cv2.medianBlur(roi, 5)
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hsv_roi = cv2.medianBlur(hsv_roi, 5)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)





#inicio do tracking
cap = cv2.VideoCapture(0)
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
while True:
    ret, frame = cap.read()


    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = cv2.medianBlur(hsv, 5)
        mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        ret, track_window = cv2.CamShift(mask, track_window, term_criteria)

        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame, [pts], True, 255, 2)

        cv2.imshow("mask", mask)
        cv2.imshow("Frame", frame)
        cv2.imshow('Detect', imagem[y: y + h, x: x + w])
        # Para visualizar histograma
        plt.hist(roi.ravel(), 256, [0, 256]);
        plt.show()

        key = cv2.waitKey(1)
        if key == 0:
            break

cap.release()
cv2.destroyAllWindows()
