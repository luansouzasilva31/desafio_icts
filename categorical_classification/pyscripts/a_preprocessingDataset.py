import os, numpy as np, matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array


dir_images = '../cats_and_dogs_dataset/images/images'
dir_save = '../cats_and_dogs_dataset/npz'
if not os.path.exists(dir_save): os.mkdir(dir_save)


names = os.listdir(dir_images);
names = [i for i in names if i.endswith(".jpg")]
# 1. Separar por raças
#       Pega cada palavra até o ultimo "_"
greed_names = [i[0:i.rindex('_')] for i in names]

#       Atribui categorias para cada raça
categories = {}; m=0
for i in greed_names:
    if i not in categories:
        categories[i]=m
        m+=1

# 2. Importar data por raça
greed_images={}
labels={}
for i in categories:
    greed_images[i] = []; labels[i] = []
    for j in names:
        if i in j:
            img = load_img(dir_images+'/'+j, target_size=(224,224,3))
            img = img_to_array(img)
            greed_images[i].extend([img])
            labels[i].append(categories[i])

# 3. Convertendo as categorias para array
for i in greed_images:
    greed_images[i] = np.asarray(greed_images[i])
    labels[i] = np.asarray(labels[i])

# 4. Dividindo dados para treino e teste
p = 0.7 # porcentagem para treino
q = 0.1 # porcentagem para validação

first=True
for i in greed_images:
    if first: # é o primeiro loop
        first = False
        images_train = greed_images[i][:int(p*greed_images[i].shape[0])]
        images_val =  greed_images[i][int(p*greed_images[i].shape[0]):int((p+q)*greed_images[i].shape[0])]
        images_test = greed_images[i][int((p+q)*greed_images[i].shape[0]):]
        
        labels_train = labels[i][:int(p*labels[i].shape[0])]
        labels_val =  labels[i][int(p*labels[i].shape[0]):int((p+q)*labels[i].shape[0])]
        labels_test = labels[i][int((p+q)*labels[i].shape[0]):]
        
    else: # não é o primeiro loop
        images_train = np.concatenate((images_train, greed_images[i][:int(p*greed_images[i].shape[0])]))
        images_val = np.concatenate((images_val, greed_images[i][int(p*greed_images[i].shape[0]):int((p+q)*greed_images[i].shape[0])]))
        images_test = np.concatenate((images_test, greed_images[i][int((p+q)*greed_images[i].shape[0]):]))

        labels_train = np.concatenate((labels_train, labels[i][:int(p*labels[i].shape[0])]))
        labels_val = np.concatenate((labels_val, labels[i][int(p*labels[i].shape[0]):int((p+q)*labels[i].shape[0])]))
        labels_test = np.concatenate((labels_test, labels[i][int((p+q)*labels[i].shape[0]):]))


# 5. Salvando data
print('Salvando data de treino, validação e teste.')
print('It will take some minutes...')
np.savez_compressed(dir_save + '/images_train', images_train)
np.save(dir_save + '/labels_train.npy', labels_train)

np.savez_compressed(dir_save + '/images_val', images_val)
np.save(dir_save + '/labels_val.npy', labels_val)

np.savez_compressed(dir_save + '/images_test', images_test)
np.save(dir_save + '/labels_test.npy', labels_test)

print('Salvamento concluído!')

