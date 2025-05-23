from tensorflow.keras import layers, models # type: ignore

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

import matplotlib.pyplot as plt



def initialize_and_prepare_information(info_dir, img_height=224, img_width=224, batch_size=32):

    """

    Initialize and prepare data information from directory

    """

    data_augmentor = ImageDataGenerator(

        rescale=1./255,

        rotation_range=20,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True,

        validation_split=0.2

    )



    data_loader = data_augmentor.flow_from_directory(

        info_dir,

        target_size=(img_height, img_width),

        batch_size=batch_size,

        class_mode='categorical',

        subset='training'

    )



    data_validator = data_augmentor.flow_from_directory(

        info_dir,

        target_size=(img_height, img_width),

        batch_size=batch_size,

        class_mode='categorical',

        subset='validation'

    )



    return data_loader, data_validator



def establish_blueprint(num_classes):

    """

    Establish a CNN blueprint for image classification

    """

    schema = models.Sequential([

        layers.Conv2D(32, 3, activation='relu', input_shape=(224, 224, 3)),

        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),

        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation='relu'),

        layers.MaxPooling2D(),

        layers.Flatten(),

        layers.Dense(64, activation='relu'),

        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')

    ])



    return schema



def educate_blueprint(schema, data_loader, data_validator, epochs=10):

    """

    Educate the blueprint

    """

    schema.compile(

        optimizer='adam',

        loss='categorical_crossentropy',

        metrics=['accuracy']

    )



    past = schema.fit(

        data_loader,

        validation_data=data_validator,

        epochs=epochs

    )



    return past



def illustrate_training_past(past):

    """

    Illustrate training past

    """

    acc = past.history['accuracy']

    val_acc = past.history['val_accuracy']

    loss = past.history['loss']

    val_loss = past.history['val_loss']



    epochs_range = range(len(acc))



    plt.figure(figsize=(12, 4))

    

    plt.subplot(1, 2, 1)

    plt.plot(epochs_range, acc, label='Training Accuracy')

    plt.plot(epochs_range, val_acc, label='Validation Accuracy')

    plt.title('Training and Validation Accuracy')

    plt.legend()



    plt.subplot(1, 2, 2)

    plt.plot(epochs_range, loss, label='Training Loss')

    plt.plot(epochs_range, val_loss, label='Validation Loss')

    plt.title('Training and Validation Loss')

    plt.legend()



    plt.show()



def initiate():

    # Example usage

    info_dir = 'path/to/image/dataset'

    num_classes = 10  # Update based on your dataset

    

    # Initialize and prepare data

    data_loader, data_validator = initialize_and_prepare_information(info_dir)

    

    # Establish and educate blueprint

    schema = establish_blueprint(num_classes)

    past = educate_blueprint(schema, data_loader, data_validator)

    

    # Illustrate results

    illustrate_training_past(past)

    

    # Save blueprint

    schema.save('image_classifier.h5')



if __name__ == "__main__":

    initiate()