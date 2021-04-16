import sys, numpy as np
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

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

# Função para plotar as métricas
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
    
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()

def executar(dir_data, dir_save):
    # Definindo modelo
    model = model_transferLearning()
    
    # Criando data gen para augmentation
    train_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, 
                                    height_shift_range=0.1, horizontal_flip=True,
                                    zoom_range=0.2, rotation_range=40,
                                    shear_range=0.2,fill_mode='nearest')
    val_datagen = ImageDataGenerator(rescale=1.0/255.0, width_shift_range=0.1, 
                                    height_shift_range=0.1, horizontal_flip=True,
                                    zoom_range=0.2, rotation_range=40,
                                    shear_range=0.2, fill_mode='nearest')    
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    
    # Importando data e separando em treino e teste
    print('\n\nImportando base de dados: train, val, test...\n')
    x_train = np.load(dir_data + '/images_train.npz'); x_train = x_train['arr_0']
    y_train = np.load(dir_data + '/labels_train.npy')
    x_val = np.load(dir_data + '/images_val.npz'); x_val = x_val['arr_0']
    y_val = np.load(dir_data + '/labels_val.npy')
    x_test = np.load(dir_data + '/images_test.npz'); x_test = x_test['arr_0']
    y_test = np.load(dir_data + '/labels_test.npy')
    
    
    # Preparando iterators
    train_it = train_datagen.flow(x_train, y_train, batch_size=32,shuffle=True)
    val_it = val_datagen.flow(x_val, y_val, batch_size = 32, shuffle = True)
    test_it = test_datagen.flow(x_test, y_test, batch_size=1)
    	
    # Definindo model checkpoint
    model_checkpoint = ModelCheckpoint(dir_save+'/model.h5', monitor='val_accuracy', save_best_only=True,
                                   verbose=1, save_weights_only=True, mode='auto')
    callbacks=[model_checkpoint]
    # Treinando modelo
    print('\n\nIniciando treinamento...\n')
    history = model.fit_generator(train_it,validation_data=val_it, validation_steps=len(val_it)//32,
                                  steps_per_epoch=None, epochs=50, verbose=1,callbacks=callbacks)
    #model.save(dir_save+'/model_3_last.h5')    
    # Avaliando modelo
    print('\n\nAvaliando modelo na base de TESTE...\n')
    _, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
    print('\n\n\t Accuracy resultante: %.3f.\n\n' % (acc * 100.0))
    print('execução finalizada.')
    #summarize_diagnostics(history)
    
dir_data = '../cats_and_dogs_dataset/npz'
dir_save = '../weights'

executar(dir_data, dir_save)

