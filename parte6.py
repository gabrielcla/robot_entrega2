# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import time

print("Use 'q' para sair e 'p' para pausar\n")


def find_homography_draw_box(kp1, kp2, img_cena):  
    out = img_cena.copy()
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    # Tenta achar uma trasformacao composta de rotacao, translacao e escala que situe uma imagem na outra
    # Esta transformação é chamada de homografia 
    # Para saber mais veja 
    # https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    h,w = img_original.shape
    
    # Um retângulo com as dimensões da imagem original
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

    # Transforma os pontos do retângulo para onde estao na imagem destino usando a homografia encontrada
    dst = cv2.perspectiveTransform(pts,M)

    # Desenha um contorno em vermelho ao redor de onde o objeto foi encontrado
    img2b = cv2.polylines(out,[np.int32(dst)],True,(255,255,0),5, cv2.LINE_AA)
    
    return img2b
    

# Carrega o video
cap = cv2.VideoCapture('sample6.mp4')

# Parâmetros pra quando usar a webcam
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)



# Identificando
img_bgr = cv2.imread('pattern1.png')
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

original_bgr = img_bgr

brisk = cv2.BRISK_create() # Nota: numa versão anterior era a BRISK
kpts = brisk.detect(img_gray)
x = [k.pt[0] for k in kpts]
y = [k.pt[1] for k in kpts]
# s will correspond to the neighborhood area
s = [(k.size/2)**2 * np.pi for k in kpts]

# Número mínimo de pontos correspondentes
MIN_MATCH_COUNT = 5

# Versões RGB das imagens, para plot
original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)


while(True):
    ret, frame = cap.read()

# 1) MANIPULAÇÕES NA IMAGEM
    # Converte o frame pra em escala de cinza
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Versões grayscale para feature matching
    img_original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    img_cena = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    framed = None
    cena_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Imagem de saída
    out = cena_rgb.copy()

    # Cria o detector BRISK
    brisk = cv2.BRISK_create()

    # Encontra os pontos únicos (keypoints) nas duas imagems
    kp1, des1 = brisk.detectAndCompute(img_original ,None)
    kp2, des2 = brisk.detectAndCompute(img_cena,None)

    # Configura o algoritmo de casamento de features que vê *como* o objeto que deve ser encontrado aparece na imagem
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Tenta fazer a melhor comparacao usando o algoritmo
    matches = bf.knnMatch(des1,des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)


    if len(good) > MIN_MATCH_COUNT:
        # Separa os bons matches na origem e no destino
        print("Matches found")    
        framed = find_homography_draw_box(kp1, kp2, cena_rgb)
        img3 = cv2.drawMatches(original_rgb,kp1,cena_rgb,kp2, good,       None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        img3_bgr = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
        cv2.imshow('Identificador', img3_bgr)

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        cv2.imshow('Identificador', frame)



# 3) TRATA TECLAS PRESSIONADAS
    # Aguarda T ms por uma tecla 
    tecla = cv2.waitKey(25)

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