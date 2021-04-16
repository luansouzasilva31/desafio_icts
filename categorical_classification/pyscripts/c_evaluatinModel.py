import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.utils.np_utils import to_categorical 
from keras.callbacks import ModelCheckpoint
from keras.applications.xception import Xception

def model_cnn(img_shape):
    base_model = Xception(input_shape=img_shape, weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)

    x = Dense(100, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(37, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.summary()

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

    return model


def evaluating(dir_data, weight_dir):
    # Criando modelo
    img_shape = (224,224,3)
    model = model_cnn(img_shape)
    
    #Importando pesos
    model.load_weights(weight_dir + '/model_last.h5')

    #Importando base de teste
    print('Importando dataset...')
    x_train = np.load(dir_data + '/images_train.npz'); x_train = x_train['arr_0']
    y_train = np.load(dir_data + '/labels_train.npy')
    y_train = to_categorical(y_train, num_classes=37)

    x_val = np.load(dir_data + '/images_val.npz'); x_val = x_val['arr_0']
    y_val = np.load(dir_data + '/labels_val.npy')
    y_val = to_categorical(y_val, num_classes=37)

    x_test = np.load(dir_data + '/images_test.npz'); x_test = x_test['arr_0']
    y_test = np.load(dir_data + '/labels_test.npy')
    y_test = to_categorical(y_test, num_classes=37)
    
    # Criando generator para normalizar dataset
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    # Aplicando generator no dataset
    train_it = datagen.flow(x_train, y_train, batch_size=1)
    val_it = datagen.flow(x_val, y_val, batch_size=1)    
    test_it = datagen.flow(x_test, y_test, batch_size=1)
    
    # Avaliando modelo
    # Não é utilizado confusion matrix pois há muitas categorias.
    print('\n\nAvaliando modelo...\n')
    _, acc = model.evaluate_generator(train_it, steps=len(train_it), verbose=1)
    print('\n\n Avaliação na base de Train:\n > %.3f de acc.\n\n' % (acc* 100.0))

    _, acc = model.evaluate_generator(val_it, steps=len(val_it), verbose=1)
    print('\n\n Avaliação na base de Val:\n > %.3f de acc.\n\n' % (acc * 100.0))

    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print('\n\n Avaliação na base de Teste:\n > %.3f de acc.\n\n' % (acc * 100.0))

dir_data = '../cats_and_dogs_dataset/npz'
weight_dir = '../weights'

evaluating(dir_data, weight_dir)







