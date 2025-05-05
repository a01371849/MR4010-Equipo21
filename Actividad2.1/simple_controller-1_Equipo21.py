"""camera_pid controller."""

# Importación de librerías necesarias
from controller import Display, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
import math

# Función para obtener una imagen de la cámara
def get_image(camera):
    # Captura la imagen en formato raw
    raw_image = camera.getImage()  
    # Convierte la imagen a un array numpy
    image = np.frombuffer(raw_image, np.uint8).reshape((camera.getHeight(), 
                                                        camera.getWidth(), 4))  
    # Extrae los canales BGR (ignora el canal alfa)
    bgr_image = image[:, :, :3]  
    # Devuelve la imagen en formato BGR
    return bgr_image  

# Función para procesar la imagen
def Procesamiento(image):
    # Conversión de la imagen a escala de grises
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar un filtro Gaussiano para reducir el ruido
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    # Detectar bordes usando el algoritmo de Canny
    edges = cv2.Canny(blur, 20, 30, apertureSize=3)
    # Obtener las dimensiones de la imagen
    altura, ancho = edges.shape
    # Crear una máscara negra del tamaño de la imagen
    mascara = np.zeros_like(edges)
    # Definir los vértices de un polígono para la detección de líneas
    vertices = np.array([[(3 * ancho // 18, altura), 
                          (15 * ancho // 18, altura), 
                          (3 * ancho // 5, 7 * altura // 10), 
                          (2 * ancho // 5, 7 * altura // 10)]], 
                          dtype=np.int32)
    # Dibujar el polígono en la máscara
    recuadro = cv2.fillPoly(mascara, vertices, 255)
    # Extraer solo la información dentro del polígono
    recorte = cv2.bitwise_and(edges, recuadro)
    # Detectar líneas usando la transformada de Hough
    lineas = cv2.HoughLinesP(recorte, rho=1, theta=np.pi/180, 
                             threshold=20, minLineLength=15, maxLineGap=20)
    # Crear una imagen en blanco para dibujar las líneas detectadas
    imagen_blanca = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    # Variable para almacenar el promedio de las coordenadas x de las líneas
    x_avg = None  
    if lineas is not None:
        sumx2 = 0
        count = 0
        for linea in lineas:
            for x1, y1, x2, y2 in linea:
                dx = x2 - x1
                dy = y2 - y1
                # Calcular el ángulo de inclinación de la línea
                if abs(dx) < 1e-3:
                    angle_deg = 90
                else:
                    m = dy / dx
                    angle_rad = math.atan(m)
                    angle_deg = abs(math.degrees(angle_rad))
                # Filtrar líneas con inclinación mayor o igual a 30 grados
                if angle_deg >= 30:
                    cv2.line(imagen_blanca, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    sumx2 += x2
                    count += 1
        if count > 0:
            x_avg = sumx2 / count  # Calcular el promedio de las coordenadas x
        # Combinar la imagen original con las líneas detectadas
        imagen = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                 0.8, imagen_blanca, 1, 0.0)
    else:
        # Si no se detectan líneas, devolver la imagen original
        imagen = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Devolver la imagen procesada y el promedio de las coordenadas x
    return imagen, x_avg  

# Función para mostrar la imagen en el display
def display_image(display, image):
    # Convertir la imagen a formato RGB
    image_rgb = image
    # Crear una referencia de imagen para el display
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    # Mostrar la imagen en el display
    display.imagePaste(image_ref, 0, 0, False)

# Variables iniciales para el ángulo y la velocidad
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 60

# Función para establecer la velocidad
def set_speed(kmh):
    global speed
    speed = kmh  # Actualiza la velocidad global

# Función para establecer el ángulo de dirección
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    steering_angle = wheel_angle
    angle = wheel_angle

# Función principal
def main():
    # Crear una instancia del robot
    robot = Car()
    driver = Driver()

    # Obtener el tiempo de paso del mundo actual
    timestep = int(robot.getBasicTimeStep())

    # Crear una instancia de la cámara
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # Habilitar la cámara con el tiempo de paso

    # Crear una instancia del display para mostrar imágenes procesadas
    display_img = Display("display_image")

    mitad = 128  # Centro de la imagen
    while robot.step() != -1:
        # Obtener la imagen de la cámara
        image = get_image(camera)

        # Procesar y mostrar la imagen
        imagen_procesada, line_avg = Procesamiento(image)
        display_image(display_img, imagen_procesada)
            
        if line_avg is not None:
            center_x = mitad
            # Calcular el error entre el centro de la imagen y la línea detectada
            error = line_avg - center_x
            # Control proporcional para el ángulo de dirección
            kP = 0.005  # Constante proporcional
            steering = error * kP
            # Limitar el ángulo de dirección
            if steering > 0.5:
                steering = 0.5
            elif steering < -0.5:
                steering = -0.5
            set_steering_angle(steering)  # Ajustar el ángulo de dirección
        else:
            # Si no se detecta una línea, mantener el vehículo recto
            set_steering_angle(0)    

        # Actualizar el ángulo y la velocidad del vehículo
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)

# Ejecutar la función principal
if __name__ == "__main__":
    main()