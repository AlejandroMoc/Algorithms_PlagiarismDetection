from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import matplotlib.pyplot as plt

def cargar_y_procesar_datos(dir_datos, altura_img=224, ancho_img=224, tam_lote=32):
    datagen_entrenamiento = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    generador_entrenamiento = datagen_entrenamiento.flow_from_directory(
        dir_datos,
        target_size=(altura_img, ancho_img),
        batch_size=tam_lote,
        class_mode='categorical',
        subset='training'
    )

    generador_validacion = datagen_entrenamiento.flow_from_directory(
        dir_datos,
        target_size=(altura_img, ancho_img),
        batch_size=tam_lote,
        class_mode='categorical',
        subset='validation'
    )

    return generador_entrenamiento, generador_validacion

def crear_modelo(num_clases):
    modelo = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_clases, activation='softmax')
    ])

    return modelo

def entrenar_modelo(modelo, generador_entrenamiento, generador_validacion, epocas=10):
    modelo.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    historial = modelo.fit(
        generador_entrenamiento,
        validation_data=generador_validacion,
        epochs=epocas
    )

    return historial

def graficar_historial_entrenamiento(historial):
    acc = historial.history['accuracy']
    val_acc = historial.history['val_accuracy']
    loss = historial.history['loss']
    val_loss = historial.history['val_loss']

    rangos_epocas = range(len(acc))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(rangos_epocas, acc, label='Precisión de Entrenamiento')
    plt.plot(rangos_epocas, val_acc, label='Precisión de Validación')
    plt.title('Precisión de Entrenamiento y Validación')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(rangos_epocas, loss, label='Pérdida de Entrenamiento')
    plt.plot(rangos_epocas, val_loss, label='Pérdida de Validación')
    plt.title('Pérdida de Entrenamiento y Validación')
    plt.legend()

    plt.show()

def principal():
    dir_datos = 'ruta/a/dataset_de_imagenes'
    num_clases = 10  # Actualizar según tu dataset

    generador_entrenamiento, generador_validacion = cargar_y_procesar_datos(dir_datos)

    modelo = crear_modelo(num_clases)
    historial = entrenar_modelo(modelo, generador_entrenamiento, generador_validacion)

    graficar_historial_entrenamiento(historial)

    modelo.save('clasificador_de_imagenes.h5')

if __name__ == "__main__":
    principal()