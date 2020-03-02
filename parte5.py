# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

print("Use 'q' para sair e 'p' para pausar\n")

# Retorna apenas os N contornos de maior area
def N_Maiores_Contornos(contornos, N):
    # N não pode ser maior que o número de contornos
    if len(contornos) < N:
        return []
    
    # Usa uma cópia pra não alterar o original
    contornos = contornos.copy()
    A, M = [0]*N, [None]*N
    maior_i= -1
    
    # Pega os N maiores da lista
    for n in range( N ):
        for i in range( len(contornos) ):
            area = cv2.contourArea(contornos[i])      
            if area > A[n]:
                maior_i = i
                A[n],M[n] = area,contornos[i]
        # Remove o maior da lista
        if (maior_i > -1) and (maior_i < len(contornos)):
            contornos.pop(maior_i) 

    # Nenhum contorno pode ser None
    for m in M:
        if m is None:
            return []
    
    return M

# Retorna o ponto do centro de um contorno
def center_of_contour(cont):
    """ Retorna uma tupla (cx, cy) que desenha o centro do contorno"""
    M = cv2.moments(cont)
    # Usando a expressão do centróide definida em: https://en.wikipedia.org/wiki/Image_moment
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (int(cX), int(cY))

# Retorna a distância entre dois pontos em pixels
def dist_entre_pontos(pontoA, pontoB):
    xA,yA = pontoA
    xB,yB = pontoB   
    dist =(xA - xB)**2 + (yA - yB)**2 
    d = np.sqrt(dist)
    return d

# Retorna a distância da câmera ao papel
def dist_camera(h):
    f, H = 908.6, 14 # px, cm
    if (h > 0):
        D = f*H/h
        return str(round(D,2))+' cm'

# Retorna a inclinação entre a reta que os une e a horizontal
def angulo(pontoA, pontoB):
    xA,yA = pontoA
    xB,yB = pontoB
    if (abs(xA-xB) != 0):
        frac = abs(yA-yB)/abs(xA-xB)
        ang = np.arctan(frac)* 180/np.pi
    else:
        ang = 90.00
    return str(round(ang,2))+' graus'

# Retorna uma imagem com as bordas que foram detectadas
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

# Carrega o video
cap = cv2.VideoCapture('sample.mp4')

# Parâmetros pra quando usar a webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Duração de cada frame
T = 25#int(1000/30)

# Intervalo de cores
hsvCi = np.array([100, 200, 130], dtype=np.uint8)
hsvCf = np.array([120, 255, 255], dtype=np.uint8)
hsvMi = np.array([165, 150, 50], dtype=np.uint8)
hsvMf = np.array([175, 255, 255], dtype=np.uint8)


while(True):
    ret, frame = cap.read()

# 1) MANIPULAÇÕES NA IMAGEM

    # Converte o frame pra HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Retira o ruído da imagem
    blur = cv2.GaussianBlur(frame_hsv, (5,5), 0)

    # Detecta as bordas presentes na imagem
    bordas = auto_canny(blur)  



    # Cria a máscara pra reconhecer os círculos através das cores
    maskM = cv2.inRange(blur, hsvMi, hsvMf)
    maskC = cv2.inRange(blur, hsvCi, hsvCf)
    mask = maskC + maskM # Une as máscaras

    # Remove buracos menores que 10x10
    segmentado = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10)))

    # Identifica os contornos da imagem
    contornos, hierarquia = cv2.findContours(segmentado.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    



    # Cria uma nova imagem vazia
    novo = np.zeros(frame.shape, dtype=np.uint8)

    # Imprime os contornos num novo frame
    cv2.drawContours(novo, contornos, -1, [0,255,0], 3);
    
    # Converte a imagem pra cinza pra fazer o Smoothing
    novo_gray = cv2.cvtColor(novo, cv2.COLOR_BGR2GRAY)

    # Pegar os circulos dentre os contornos identificados
    circles = cv2.HoughCircles(novo_gray, cv2.HOUGH_GRADIENT, 4, 20, minRadius=5, maxRadius=100)
    if circles is not None:        
        circles = np.uint16(np.around(circles))
        circs = circles[0,:]
        for i in circs:
            cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),4) 



# 2) EXIBE O FRAME
    cv2.imshow('Detector de circulos', frame)



# 3) TRATA TECLAS PRESSIONADAS
    # Aguarda T ms por uma tecla 
    tecla = cv2.waitKey(T)

    # A tecla 'p' pausa no frame exibido até que seja pressionada outra tecla
    if tecla == ord('p'):
        print('\nPausado pelo usuário \nPressione qualquer tecla para continuar \n')
        cv2.waitKey()
        continue

    # A tecla 'q' finaliza o loop
    if tecla == ord('q'):
        print('\nFinalizado pelo usuário\n')
        break # Finaliza o loop

#  Fecha o arquivo ou dispositivo usado
cap.release()
# Destrói todas as janelas criadas
cv2.destroyAllWindows()