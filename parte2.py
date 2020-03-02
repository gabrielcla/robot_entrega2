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



# Carrega o video
cap = cv2.VideoCapture('sample.mp4')

# Parâmetros pra quando usar a webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Duração de cada frame
T = 25

# Intervalo de cores dos círculos ciano e magenta
hsvCi = np.array([100, 200, 130], dtype=np.uint8)
hsvCf = np.array([120, 255, 255], dtype=np.uint8)
hsvMi = np.array([165, 150, 50], dtype=np.uint8)
hsvMf = np.array([175, 255, 255], dtype=np.uint8)


while(True):
    ret, frame = cap.read()

# 1) MANIPULAÇÕES NA IMAGEM
    # Retira o ruído da imagem (Smoothing)
    blur = cv2.GaussianBlur(frame,(5,5),0)

    # Converte o frame pra HSV
    frame_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    
    # Cria a máscara pra reconhecer as bolas através das cores
    maskM = cv2.inRange(frame_hsv, hsvMi, hsvMf)
    maskC = cv2.inRange(frame_hsv, hsvCi, hsvCf)
    mask = maskC + maskM # Une as máscaras

    # Remove buracos menores que 10x10
    segmentado = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10)))

    # Identifica os contornos da imagem
    contornos, hierarquia = cv2.findContours(segmentado.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

    # Pega só os 2 maiores contornos
    maiores = N_Maiores_Contornos(contornos, 2)

    if len(maiores) == 2:
        # Desenha os contornos nos círculos
        cv2.drawContours(frame, maiores, -1, [0,255,0], 3); # -1 = desenha todos da lista de contornos



# 2) EXIBE O FRAME
    cv2.imshow('Parte 2', frame)



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
