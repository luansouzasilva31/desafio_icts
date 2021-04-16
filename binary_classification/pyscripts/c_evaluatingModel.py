import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator

# Definindo modelo VGGNet com 3 blocos
def model_transferLearning():
    model = VGG16(include_top=False, input_shape=(224,224,3))
    model.trainable  = True
    set_trainable = False
    for layer in model.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    #Adicionando parte sequencial
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(1, activation='sigmoid')(class1)

    model = Model(inputs=model.inputs, outputs=output)

    #opt = SGD(lr=0.001, momentum=0.9)
    #opt = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.000000199)
    opt = RMSprop(lr=1e-5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluating(dir_data, weight_dir):
    # Criando modelo
    model = model_transferLearning()
    
    #Importando pesos
    model.load_weights(weight_dir + '/model.h5')

    #Importando base de teste
    x_train = np.load(dir_data + '/images_train.npz'); x_train = x_train['arr_0']
    y_train = np.load(dir_data + '/labels_train.npy')
    
    x_val = np.load(dir_data + '/images_val.npz'); x_val = x_val['arr_0']
    y_val = np.load(dir_data + '/labels_val.npy')
    
    x_test = np.load(dir_data + '/images_test.npz'); x_test = x_test['arr_0']
    y_test = np.load(dir_data + '/labels_test.npy')
    
    # Criando generator para normalizar dataset
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Aplicando generator no dataset
    train_it = datagen.flow(x_train, y_train, batch_size=32)
    val_it = datagen.flow(x_val, y_val, batch_size=32)
    test_it = datagen.flow(x_test, y_test, batch_size=32)
    
    # Avaliando modelo
    print('\n\nAvaliando modelo...\n')
    _, acc = model.evaluate_generator(train_it, steps=len(train_it), verbose=1)
    print('Avaliação na base de Train: %.3f de acc.' % (acc * 100.0))

    _, acc = model.evaluate_generator(val_it, steps=len(val_it), verbose=1)
    print('\n Avaliação na base de Val: %.3f de acc.' % (acc * 100.0))

    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print('\n Avaliação na base de Teste: %.3f de acc.\n\n' % (acc * 100.0))

dir_data = '../cats_and_dogs_dataset/npz'
weight_dir = '../weights'

evaluating(dir_data, weight_dir)



