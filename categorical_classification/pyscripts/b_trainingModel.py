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

def executar(dir_data, dir_save):
    img_shape = (224,224,3)
    
    model = model_cnn(img_shape)
    
    # Criando data gen para augmentation
    datagen = ImageDataGenerator(rescale=1.0/255.0,horizontal_flip=True, rotation_range=15, fill_mode='nearest',zoom_range=0.2,width_shift_range=0.1,height_shift_range=0.1)
    
    # Importando data e separando em treino e teste
    print('\n\nImportando base de dados: train, val, test...\n')
    x_train = np.load(dir_data + '/images_train.npz'); x_train = x_train['arr_0']
    y_train = np.load(dir_data + '/labels_train.npy')
    x_val = np.load(dir_data + '/images_val.npz'); x_val = x_val['arr_0']
    y_val = np.load(dir_data + '/labels_val.npy')
    x_test = np.load(dir_data + '/images_test.npz'); x_test = x_test['arr_0']
    y_test = np.load(dir_data + '/labels_test.npy')
    
    # Convertendo para dados categóricos
    y_train = to_categorical(y_train, num_classes=37)
    y_val = to_categorical(y_val, num_classes=37)
    y_test = to_categorical(y_test, num_classes=37)
    
    # Preparando iterators
    train_it = datagen.flow(x_train, y_train, batch_size=8,shuffle=True)
    val_it = datagen.flow(x_val, y_val, batch_size = 8, shuffle = True)
    test_it = datagen.flow(x_test, y_test, batch_size=1)
    	
    # Definindo model checkpoint
    model_checkpoint = ModelCheckpoint(dir_save+'/model.h5', monitor='val_accuracy', save_best_only=True,
                                   verbose=1, save_weights_only=True, mode='auto')

    # Treinando modelo
    print('\n\nIniciando treinamento...\n')
    history = model.fit_generator(train_it,validation_data=val_it, epochs=50,
                                  verbose=1, callbacks=[model_checkpoint])
    model.save(dir_save+'/model_last.h5') # Salva modelo da última rodada, caso o método ModelCheckpoint falhe. Apenas garantia.

    #summarize_diagnostics(history)
    
dir_data = '../cats_and_dogs_dataset/npz'
dir_save = '../weights'

executar(dir_data, dir_save)




