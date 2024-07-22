
Operadores

+
-
*
**    potencia
/
//    división entera
%     módulo


==
!=
<
<=
>
>=


and
or
not



Concatenar datos en un print

nombre = Facundo
apellido = Gaitán
print(f"Hola, me llamo {nombre} {apellido}")



Entrada de datos

dato = input("¿?: ")

version = int(input("¿Qué versión es?: "))       input básico es string, acá se convierte a int
print(f"La versión es {version + 1}")            se puede operar con las variables dentro del {}



if dato > 0:
    print("Positivo")
elif dato == 0:
    print("Cero")
else:
    print("Negativo")



Tratar string como lista(array)

nombre[1] primera letra
nombre[-1] última letra

'''




# Ejercicio 1
# Función matemática de tres variables

a = float(input("a: "))
b = float(input("b: "))
c = float(input("c: "))

resultado = ((c+5)*(b**2-3*a*c)*a**2)/4*a

print(f"El resultado es {resultado}")



# Ejercicio 2
# Intercambiar valores de variables

a = input("a: ")
b = input("b: ")

a, b = b, a



# Ejercicio 3
# Área y perímetro del círculo

import math

r = float(input("Radio: ")) 
area = math.pi*(r**2)
perimetro = 2*math.pi*r



# Ejercicio 4
# 36% de descuento

precio = float(input("Precio: "))
descuento = precio*(36/100)
precio_con_descuento = precio-descuento       # o directamente precio*(1-0.36)

print(f"El precio final es: {precio_con_descuento:.2f}")     # dos decimales



#--------------------------------------------------------------------------------------------------------------



# Ejercicio 1
# Dos numeros, cuál es par

n1 = int(input("Primer número: "))
n2 = int(input("Segundo número: "))

if n1%2 == 0 and n2%2 == 0:
    print("Ambos son pares")
elif n1%2 == 0:
    print("El primero es par")
elif n2%2 == 0:
    print("El segundo es par")
else:
    print("Ninguno es par")




# Ejercicio 2
# Cajero automático

saldo = 2000

seleccion = int(input("Elija una opción: /n1. Ingreso de dinero/n2. Retiro de dinero/n3. Mostrar saldo/n4. Salir"))

if seleccion == 1:
    ingreso = float(input("Dinero a ingresar: "))
    saldo += ingreso
    print(f"Saldo: {saldo}")
elif seleccion == 2:
    retiro = float(input("Dinero a retirar: "))
    if retiro > saldo:
        print("Saldo insuficiente")
    else:
        saldo -= retiro
        print(f"Saldo: {saldo}")
elif seleccion == 3:
    print(f"Saldo: {saldo}")
elif seleccion == 4:
    print("Fin")
else:
    print("Error")





# Conjuntos

A = {1,2,3,4}
B = {2,3,5,6}
C = {3,4,6,7}

print(A==B)     # Comparar si son iguales
print(A|B)      # Unir conjuntos {1,2,3,4,5,6}
print(A|C)      #   "      "     {1,2,3,4,6,7}  
print(A&B)      # Interseccion de conjuntos {2,3}
print(A&C)      #      "       "      "     {3,4}
print(A-B)      # Diferencia de conjuntos (A pero no B) {1,4}
print(A^B)      # Diferencia simétrica (no toma en cuenta las intersecciones) {1,4,5,6}







# Análisis de imágenes (con OpenCv)
'''
#1 Escala de grises
#2 Umbralización (diferenciar objeto de su espacio)
#3 Segmentación y contornos


Instalar: 
"opencv contrib python" en Google
cmd y pegar el comando que sale


import cv2
print(cv2.__version__)

'''


# Encontrar contornos en una imagen
from cv2 import cv2

imagen = cv2.imread('contornos.jpg')
grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
_, umbral = cv2.threshold(grises, 100, 255, cv2.THRESH_BINARY)    #devuelve dos valores, el umbral usado y la imágen
contorno, jerarquia = cv2.findCountours(umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  
cv2.drawContours(imagen, contorno, -1, (0, 255, 0), 3)

cv2.imshow('Imagen original', imagen)
cv2.imshow('Imagen en grises', grises)
cv2.imshow('Imagen umbral', umbral)

cv2.waitKey(0)
cv2.destroyAllWindows()

#cv2.findCountours(img, mode, method)         métodos APROX_NONE(todo el borde) y APROX_SIMPLE(solo vértices)






# Eliminar ruido
# Suavizado Gaussiano
# Contador de monedas

from cv2 import cv2
import numpy as np

valorGauss = 3
valorKernel = 3

original = cv2.imread('monedas.jpg')
grises = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(grises, (valorGauss, valorGauss), 0)
canny = cv2.Canny(gauss, 60, 100)
kernel = np.ones((valorKernel, valorKernel), np.uint8)
cierre = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)

contornos, jerarquia = cv2.findCountours(cierre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

print("Monedas encontradas: {}".format(len(contornos)))
cv2.drawContours(original, contornos, -1, (0, 255, 0), 3)

cv2.imshow("Grises", grises)
cv2.imshow("Gauss", gauss)
cv2.imshow("Canny", canny)
cv2.imshow("Cierre", cierre)
cv2.imshow("Resultado", original)
cv2.waitKey(0)







# Con cámara (usar DroidCam en Playstore y en pc para conectar la cámara del celular) (ApowerMirror app)

import cv2 as cv

capturar_video = cv.VideoCapture(0)
if not capturar_video.isOpened():
    print("No se encontró cámara")
    exit()
while True:
    tipo_de_camara, camara = capturar_video.read()
    grises = cv.cvtColor(camara, cv.COLOR_BGR2GRAY)
    
    cv.imshow("En vivo", camara)
    if cv.waitKey(1) == ord("q"):   # Cierra al apretar letra q
        break
capturar_video.release()
cv.destroyAllWindows()



#-----------------------------------------------------------------------------------------------------



from cv2 import cv2
import numpy as np

def ordenarPuntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos:n_puntos[1])
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order:x1_order[0])
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order:x2_order[0])
    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def alineamiento(imagen, ancho, alto):
    imagen_alineada = None
    grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    tipo_umbral, umbral = cv2.threshold(grises, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow('Imagen umbral', umbral)
    contorno = cv2.findCountours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contorno = sorted(contorno, key=cv2.contourArea, reverse=True)[:1]
    for c in contorno:
        epsilon = 0.01*cv2.arcLength(c, True)
        aprox = cv2.approxPolyDP(c, epsilon, True)
        if len(aprox) == 4:
            puntos = ordenarPuntos(aprox)
            puntos1 = np.float32(puntos)
            puntos2 = np.float32([[0,0], [ancho,0], [0,alto],[ancho, alto]])
            M = cv2.getPerspectiveTransform(puntos1, puntos2)
            imagen_alineada = cv2.warpPerspective(imagen, M, (ancho,alto))
    return imagen_alineada

captura_video = cv2.VideoCapture(0)
while True:
    tipo_camara, camara = captura_video.read()
    if tipo_camara == False:
        break
    imagen_A6 = alineamiento(camara, ancho=480, alto=677)
    if imagen_A6 is not None:
        puntos = []
        imagen_gris = cv2.cvtColor(imagen_A6, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(imagen_gris, (5, 5), 1)
        _, umbral_2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
        cv2.imshow("Umbral", umbral_2)
        contorno_2 = cv2.findCountours(umbral_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(imagen_A6, contorno_2, -1, (255, 0, 0), 2)
        suma1 = 0.0
        suma2 = 0.0
        for c_2 in contorno_2:
            area = cv2.contourArea(c_2)
            Momentos = cv2.moments(c_2)
            if (Momentos["m00"]==0):
                Momentos["m00"] = 1.0
            x = int(Momentos["m10"]/Momentos["m00"])
            y = int(Momentos["m01"]/Momentos["m00"])
            
            if area < 9300 and area > 8000:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "$0.20", (x, y), font, 0.75, (0, 255, 0), 2)
                suma1 = suma1 + 0.2
            if area < 7800 and area > 6500:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(imagen_A6, "$0.10", (x, y), font, 0.75, (0, 255, 0), 2)
                suma2 = suma2 + 0.1
        
        total = suma1 + suma2
        print("Sumatoria total en centimos: ", round(total, 2))
        cv2.imshow('Imagen A6', imagen_A6)
        cv2.imshow('Cámara', camara)
    if cv2.waitKey(1) == ord("s"):
        break
captura_video.release()
cv2.destroyAllWindows()







#----------------------------------------------------------------------------------------------------------






# Redes neuronales artificiales
# Comparaciones entrelazadas (capa de entrada, capas ocultas y capa de salida)


# Reconocimiento facial 1

# Repositorio OpenCv de ruidos y objetos que no son rostros (github.com/opencv/opencv/tree/master/data/haarcascades)
# Usar el archivo haarcascade_frontalface_default


# capaentrada.py

import cv2 as cv
import os
import imutils

modelo = "Fotos"
ruta1 = "./reconocimientofacial"
rutacompleta = ruta1 + "/" + modelo
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)

ruidos = cv.CascadeClassifier("/haarcascade_frontalface_default.xml")   # Poner bien la ruta al archivo
camara = cv.VideoCapture(0)   # Se puede poner un video entre los paréntesis si no se tiene cámara: cv.VideoCapture("video.mp4")
id = 0
while True:
    respuesta, captura = camara.read()
    if respuesta == False:break
    captura = imutils.resize(captura, width=640)
    
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    id_captura = captura.copy()
    
    caras = ruidos.detectMultiScale(grises, 1.3, 5)
    
    for (x,y,e1,e2) in cara:
        cv.rectangle(captura, (x, y), (x+e1, y+e2), (255, 0, 0), 2)
        rostro_capturado = id_captura[y:y+e2, x:x+e1]
        rostro_capturado = cv.resize(rostro_capturado, (160, 160), interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta + "/imagen_{}.jpg".format(id), rostro_capturado)
        id = id+1
    
    cv.imshow("Resultado rostro", captura)
    
    if id > 350:
        break
    
    if cv.waitKey(1) == ord("s"):
        break
camara.release()
cv.destroyAllWindows()







# capaocultaentrenamiento.py

import cv2 as cv
import os
import numpy as np

data_ruta = "./reconocimientofacial"
lista_data = os.listdir(data_ruta)
#print(lista_data)

ids = []
rostros_data = []

id = 0
for fila in lista_data:
    ruta_completa = data_ruta + "/" + fila
    for archivo in os.listdir(ruta_completa):
        ids.append(id)
        rostros_data.append(cv.imread(ruta_completa+"/"+archivo, 0))
    id = id+1
entrenamiento_modelo_1 = cv.face.EigenFaceRecognizer_create()
print("Inicio entrenamiento")
entrenamiento_modelo_1.train(rostros_data, np.array(ids))
entrenamiento_modelo_1.write("Entrenamiento_EigenFaceRecognizer.xml")
print("Entrenamiento concluido")








# capasalidarecfacial.py

import cv2 as cv
import os
import imutils

data_ruta = "./reconocimientofacial"
lista_data = os.listdir(data_ruta)

entrenamiento_modelo_1 = cv.face.EigenFaceRecognizer_create()
entrenamiento_modelo_1.read("./Entrenamiento_EigenFaceRecognizer.xml")
ruidos = cv.CascadeClassifier("/haarcascade_frontalface_default.xml")    # Poner bien la ruta al archivo
camara = cv.VideoCapture(0)
while True:
    respuesta, captura = camara.read()
    if respuesta == False:break
    captura = imutils.resize(captura, width=640)
    
    grises = cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    id_captura = grises.copy()
    cara = ruidos.detectMultiScale(grises, 1.3, 5)
    for (x,y,e1,e2) in cara:
        #cv.rectangle(captura, (x, y), (x+e1, y+e2), (255, 0, 0), 2)
        rostro_capturado = id_captura[y:y+e2, x:x+e1]
        rostro_capturado = cv.resize(rostro_capturado, (160, 160), interpolation=cv.INTER_CUBIC)
        resultado = entrenamiento_modelo_1.predict(rostro_capturado)
        cv.putText(captura, '{}'.format(resultado), (x,y-5), 1, 1.3, (0,255,0), 1, cv.LINE_AA)
        
        if resultado[1]<9000:
            cv.putText(captura, '{}'.format(lista_data[resultado[0]]), (x,y-20), 1, 1.3, (0,255,0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, y+e2), (0,255,0), 2)
        else:
            cv.putText(captura, "No encontrado", (x,y-20), 1, 1.3, (0,255,0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1, y+e2), (0,255,0), 2)
        
    cv.imshow("Resultados", captura)
    if cv.waitKey(1) == ord("s"):
        break
camara.release()
cv.destroyAllWindows()