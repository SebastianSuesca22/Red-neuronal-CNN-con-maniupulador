# -*- coding: utf-8 -*-

try:
    import sim  # Importar la API remota de CoppeliaSim
except:
    print('--------------------------------------------------------------')
    print('"sim.py" could not be imported. This means very probably that')
    print('either "sim.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "sim.py"')
    print('--------------------------------------------------------------')
    print('')

import time
import numpy as np
import cv2
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tkinter import *


# PARTE 1 - Construir el modelo de RCNN
classifier = Sequential()

# Paso 1 - Convoluci�n
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(256, 256, 3), activation="relu"))

# Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Una segunda capa de convoluci�n y max pooling
classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Paso 3 - Flattening
classifier.add(Flatten())

# Paso 4 - Full Connection
classifier.add(Dense(units=128, activation="relu"))

# Capa de salida
classifier.add(Dense(units=5, activation="softmax"))  # 5 clases

# Compilar la CNN
classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# PARTE 2 - Ajustar la CNN a las im�genes para entrenar
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory(
    'dataset/entrenamiento',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'  # Cambiado a 'categorical' para 5 clases
)

testing_dataset = test_datagen.flow_from_directory(
    'dataset/prueba',
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'  # Cambiado a 'categorical' para 5 clases
)

classifier.fit(
    training_dataset,
    steps_per_epoch=training_dataset.samples // 32,
    epochs=20,
    validation_data=testing_dataset,
    validation_steps=testing_dataset.samples // 32
)

global con_caja

bandera = 1
con_caja = 0


def Leer_camara(sim, clientID, visionSensorHandle, pauseTime):

    # Pausar por un tiempo especificado
    time.sleep(pauseTime)

    # Iniciar la transmisión de datos de la cámara
    returnCode, resolution, imageBuffer = sim.simxGetVisionSensorImage(clientID, visionSensorHandle, 0, sim.simx_opmode_streaming)
    time.sleep(0.1)  # Pequeña pausa para permitir la transmisión

    # Intentar obtener la imagen del buffer
    returnCode, resolution, imageBuffer = sim.simxGetVisionSensorImage(clientID, visionSensorHandle, 0, sim.simx_opmode_buffer)

    # Verificar si la imagen fue capturada correctamente
    if returnCode == sim.simx_return_ok:
        # Transformar el buffer de imagen en una matriz de imagen
        image = np.array(imageBuffer, dtype=np.uint8)
        image = image.reshape((resolution[1], resolution[0], 3))
        image = np.flipud(image)  # Invertir la imagen verticalmente para que coincida con MATLAB
        return image 
    else:
        raise Exception('Error al obtener la imagen del sensor de visión.')

def MoverBrazo(sim, clientID, sixJoints, t1, t2, t3, t4, t5, t6):
    # Convertir ángulos a radianes
    jointAngles = [t1 * (math.pi / 180), t2 * (math.pi / 180), t3 * (math.pi / 180), 
                   t4 * (math.pi / 180), t5 * (math.pi / 180), t6 * (math.pi / 180)]

    # Pausar la comunicación para enviar múltiples comandos al mismo tiempo
    sim.simxPauseCommunication(clientID, True)
    
    # Establecer las posiciones de destino para cada junta
    for i in range(6):
        # Asegúrate de que sixJoints[i] sea un entero
        sim.simxSetJointTargetPosition(clientID, int(sixJoints[i]), jointAngles[i], sim.simx_opmode_oneshot)
    
    # Reanudar la comunicación
    sim.simxPauseCommunication(clientID, False)

    # Pausar un poco para asegurarse de que los comandos se envíen correctamente
    time.sleep(0.02)

def Detectar_Imagen(imagen):
    # Predecir una sola imagen
    test_image = img_to_array(imagen) # Convertir la imagen a un arreglo numpy
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 256.0  # Normalizar la imagen


    result = classifier.predict(test_image)

    # Obtener las clases del conjunto de entrenamiento
    class_indices = training_dataset.class_indices
    class_labels = list(class_indices.keys())
    predicted_class = class_labels[np.argmax(result)]

    #print(f'La imagen es de la clase: {predicted_class}')

    return predicted_class

def actualizar_destino():
    global destino
    destino = str(seleccion.get())  # Guarda el número seleccionado como cadena

def iniciar_simulacion():

    global bandera
    global con_caja
    global destino

    print('Program started')

    # Cerrar todas las conexiones abiertas en caso de que existan
    sim.simxFinish(-1)

    # Conectar a CoppeliaSim
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)

    if clientID != -1:
        print('Connected to remote API server')

        # Nombres de las juntas del UR5
        JointNames = ['UR5_joint1', 'UR5_joint2', 'UR5_joint3', 'UR5_joint4', 'UR5_joint5', 'UR5_joint6']

        # Inicializar array para almacenar los identificadores de las juntas
        sixJoints = np.zeros(6)

        # Iniciar la simulación
        res1 = sim.simxStartSimulation(clientID, sim.simx_opmode_oneshot_wait)
        if res1 == sim.simx_return_ok:
            print('Simulation started successfully')

            # Obtener los manejadores de las juntas y el sensor de visión
            for i in range(6):
                res1, joint_handle = sim.simxGetObjectHandle(clientID, JointNames[i], sim.simx_opmode_blocking)
                if res1 == sim.simx_return_ok:
                    sixJoints[i] = joint_handle
                else:
                    print(f'Failed to get handle for {JointNames[i]}')

            res1, visionSensorHandle = sim.simxGetObjectHandle(clientID, 'Vision_sensor', sim.simx_opmode_blocking)
            if res1 == sim.simx_return_ok:
                print('Vision sensor handle obtained successfully')
            else:
                print('Failed to get vision sensor handle')


            if con_caja==0:
                MoverBrazo(sim, clientID, sixJoints, 95, 70, 0, 0, -90, 0)
                time.sleep(4)
                MoverBrazo(sim, clientID, sixJoints, 95, 70, 20, 0, -90, 0)
                time.sleep(2)
                MoverBrazo(sim, clientID, sixJoints, 95, 70, 0, 0, -90, 0)
                time.sleep(2)
                con_caja=1
            elif con_caja==1:
                MoverBrazo(sim, clientID, sixJoints, 89, 70, 0, 0, -90, 0)
                time.sleep(4)
                MoverBrazo(sim, clientID, sixJoints, 89, 70, 20, 0, -90, 0)
                time.sleep(2)
                MoverBrazo(sim, clientID, sixJoints, 89, 70, 0, 0, -90, 0)
                time.sleep(2)
                con_caja=2
            elif con_caja==2:
                MoverBrazo(sim, clientID, sixJoints, 80, 75, -10, 20, -90, 0)
                time.sleep(4)
                MoverBrazo(sim, clientID, sixJoints, 80, 79, 0, 20, -90, 0)
                time.sleep(2)
                MoverBrazo(sim, clientID, sixJoints, 80, 75, -10, 20, -90, 0)
                time.sleep(2)
                con_caja=3
            elif con_caja==3:
                MoverBrazo(sim, clientID, sixJoints, 71, 30, 4, 10, -90, 0)
                time.sleep(4)
                MoverBrazo(sim, clientID, sixJoints, 71, 78, 4, 10, -90, 0)
                time.sleep(2)
                MoverBrazo(sim, clientID, sixJoints, 71, 30, 4, 10, -90, 0)
                time.sleep(2)
                con_caja=4
            elif con_caja==4:
                MoverBrazo(sim, clientID, sixJoints, 95, 45, 8, -20, -90, 0)
                time.sleep(4)
                MoverBrazo(sim, clientID, sixJoints, 95, 45, 68, -20, -90, 0)
                time.sleep(3)
                MoverBrazo(sim, clientID, sixJoints, 95, 45, 8, -20, -90, 0)
                time.sleep(3)
                con_caja=5
            elif con_caja==5:
                MoverBrazo(sim, clientID, sixJoints, 87, 41, 6, -25, -90, 0)
                time.sleep(4)
                MoverBrazo(sim, clientID, sixJoints, 87, 41, 76, -25, -90, 0)
                time.sleep(3)
                MoverBrazo(sim, clientID, sixJoints, 87, 41, 6, -25, -90, 0)
                time.sleep(3)
                con_caja=6


            if bandera==0:
                bandera=bandera+1
            
            if bandera==1:
                MoverBrazo(sim, clientID, sixJoints, -20, -60, 110, 30, -90, 0)
                time.sleep(4)
                image = Leer_camara(sim, clientID, visionSensorHandle, 1)
                numero = Detectar_Imagen(image)
                #numero = 1
                print(numero)

                if numero == destino:
                    MoverBrazo(sim, clientID, sixJoints, -10, 60, 8, 20, -90, 0)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 0, sim.simx_opmode_oneshot)
                    time.sleep(4)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    time.sleep(2)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    bandera=0
                    
                bandera=bandera+1

            if bandera==2:
                MoverBrazo(sim, clientID, sixJoints, -58, 0, 70, 30, -90, 0)
                time.sleep(5)
                image = Leer_camara(sim, clientID, visionSensorHandle, 1)
                numero = Detectar_Imagen(image)
                #numero = 2
                print(numero)
                if numero == destino:
                    MoverBrazo(sim, clientID, sixJoints, -70, 60, 8, 20, -90, 0)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 0, sim.simx_opmode_oneshot)
                    time.sleep(4)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    time.sleep(2)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    bandera=0

                bandera=bandera+1

            if bandera==3:
                MoverBrazo(sim, clientID, sixJoints, -105, 0, 70, 44, -90, 0)
                time.sleep(4)
                image = Leer_camara(sim, clientID, visionSensorHandle, 1)
                numero = Detectar_Imagen(image)
                #numero = 3
                print(numero)
                if numero == destino:
                    MoverBrazo(sim, clientID, sixJoints, -100, 60, 8, 20, -90, 0)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 0, sim.simx_opmode_oneshot)
                    time.sleep(4)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    time.sleep(2)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    bandera=0

                bandera=bandera+1

            if bandera==4:
                MoverBrazo(sim, clientID, sixJoints, -145, 0, 60, 40, -90, 0)
                time.sleep(4)
                image = Leer_camara(sim, clientID, visionSensorHandle, 1)
                numero = Detectar_Imagen(image)
                #numero = 4
                print(numero)
                if numero == destino:
                    MoverBrazo(sim, clientID, sixJoints, -125, 60, 8, 20, -90, 0)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 0, sim.simx_opmode_oneshot)
                    time.sleep(4)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    time.sleep(2)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    bandera=0

                bandera=bandera+1

            if bandera==5:
                MoverBrazo(sim, clientID, sixJoints, -192, -20, 70, 48, -90, 0)
                time.sleep(4)
                image = Leer_camara(sim, clientID, visionSensorHandle, 1)
                numero = Detectar_Imagen(image)
                #numero = 5
                print(str(numero) + "=" + destino)
                if numero == destino:
                    print("Ingreso...")
                    MoverBrazo(sim, clientID, sixJoints, -185, 60, 8, 20, -90, 0)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 0, sim.simx_opmode_oneshot)
                    time.sleep(4)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    time.sleep(2)
                    sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                    bandera=0

                bandera=bandera+1
                
            if bandera >5:
                MoverBrazo(sim, clientID, sixJoints, 140, 53, 0, 0, -90, 0)
                sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 0, sim.simx_opmode_oneshot)
                time.sleep(10)
                sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                time.sleep(2)
                sim.simxSetIntegerSignal(clientID, 'releaseSuctionPad', 1, sim.simx_opmode_oneshot)
                bandera=0
            

            time.sleep(2)
            MoverBrazo(sim, clientID, sixJoints, 0, 0, 0, 0, 0, 0)
            time.sleep(4)

            # Detener la simulación (opcional)
            #sim.simxStopSimulation(clientID, sim.simx_opmode_oneshot)
            sim.simxGetPingTime(clientID)  # Asegurarse de que todos los comandos hayan llegado a destino

        else:
            print('Failed to start simulation')

        # Cerrar la conexión con CoppeliaSim
        sim.simxFinish(clientID)
    else:
        print('Failed connecting to remote API server')

    print('Program ended')


raiz = Tk()
raiz.geometry("300x400")


lbl6 = Label(raiz, text="Seleccione el destino de la caja: ", anchor='w')
lbl6.place(x=10, y=100, width=200, height=30)

R_1 = Label(raiz, anchor='w')
R_1.place(x=220, y=100, width=100, height=30)
R_1.config(text=f"{con_caja+1}")

# Variable para almacenar el valor del radiobutton seleccionado
seleccion = IntVar()

lbl6 = Label(raiz, text="Seleccione el destino de la caja: ", anchor='w')
lbl6.place(x=10, y=100, width=200, height=30)

R_1 = Label(raiz, anchor='w')
R_1.place(x=220, y=100, width=100, height=30)

# Creación de los botones de radio
radio1 = Radiobutton(raiz, text="Tamaño 1", variable=seleccion, value=1, command=actualizar_destino)
radio1.place(x=10, y=150)

radio2 = Radiobutton(raiz, text="Tamaño 2", variable=seleccion, value=2, command=actualizar_destino)
radio2.place(x=10, y=180)

radio3 = Radiobutton(raiz, text="Tamaño 3", variable=seleccion, value=3, command=actualizar_destino)
radio3.place(x=10, y=210)

radio4 = Radiobutton(raiz, text="Tamaño 4", variable=seleccion, value=4, command=actualizar_destino)
radio4.place(x=10, y=240)

radio5 = Radiobutton(raiz, text="Tamaño 5", variable=seleccion, value=5, command=actualizar_destino)
radio5.place(x=10, y=270)

# Botón para iniciar la simulación
start_button = Button(raiz, text="Iniciar Simulación", command=iniciar_simulacion)
start_button.place(x=10, y=320)

raiz.mainloop()