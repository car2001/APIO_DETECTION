# scripts/preprocess.py
import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def resize_images(input_dir, output_dir, size=(224, 224)):
    """Redimensiona las imágenes y las guarda en un directorio especificado."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = cv2.imread(os.path.join(input_dir, filename))
            resized_img = cv2.resize(img, size)
            cv2.imwrite(os.path.join(output_dir, filename), resized_img)

def create_image_generators(train_dir, test_dir, target_size=(224, 224), batch_size=32):
    """Crea generadores de imágenes con normalización y aumentación de datos."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalización
        shear_range=0.2,  # Transformaciones angulares
        zoom_range=0.2,   # Aplicar zoom
        horizontal_flip=True  # Inversión horizontal
    )

    test_datagen = ImageDataGenerator(rescale=1./255)  # Solo normalización para el conjunto de prueba

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    # test_generator = test_datagen.flow_from_directory(
    #     test_dir,
    #     target_size=target_size,
    #     batch_size=batch_size,
    #     class_mode='binary'
    # )

    # return train_generator, test_generator
    return train_generator

if __name__ == "__main__":
    current_directory = os.getcwd()
    # Ruta de imágenes originales
    raw_dir = f'{current_directory}\\data\\raw_images\\'
    # Ruta de imágenes preprocesadas
    preprocessed_dir = f'{current_directory}\\data\\preprocessed_images\\'

    # Redimensionar imágenes
    # resize_images(os.path.join(raw_dir, 'con_plaga'), os.path.join(preprocessed_dir, 'con_plaga'))
    # resize_images(os.path.join(raw_dir, 'sin_plaga'), os.path.join(preprocessed_dir, 'sin_plaga'))

    # Crear generadores de imágenes
    # train_gen, test_gen = create_image_generators(preprocessed_train_dir, preprocessed_test_dir)
    train_gen = create_image_generators(preprocessed_dir, "")
    print("Preprocesamiento completado y generadores creados.")
