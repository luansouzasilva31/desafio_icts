import os, numpy as np, matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array

dir_images = '../cats_and_dogs_dataset/images/images'
dir_save = '../cats_and_dogs_dataset/npz'
if not os.path.exists(dir_save): os.mkdir(dir_save)


names = os.listdir(dir_images)

# 1. Separar nomes para gatos e cachorros, baseado no padrão de nomenclatura.
#       Fazer as duas condições evita incorporar dados errôneos, caso algum
#       nome esteja fora do padrão.
cat_names = [i for i in names if (ord(i[0])>=65 and ord(i[0])<=90  and i.endswith(".jpg"))]
dog_names = [i for i in names if (ord(i[0])>=97 and ord(i[0])<=122 and i.endswith(".jpg"))]

print('\n\n %d arquivos foram ignorados. Extensão incoerente. \n\n'%(len(names) - (len(cat_names) + len(dog_names))))

# 2. Importando as imagens
#cat_images = [cv2.imread(dir_images+'/'+i) for i in cat_names]
#dog_images = [cv2.imread(dir_images+'/'+i) for i in dog_names]
cat_images = []; dog_images = []
cat_labels = []; dog_labels = []
for name in cat_names:
    img = load_img(dir_images+'/'+name, target_size=(224, 224))
    img = img_to_array(img)
    cat_images.append(img)
    cat_labels.append(0)
for name in dog_names:
    img = load_img(dir_images+'/'+name, target_size=(224, 224))
    img = img_to_array(img)
    dog_images.append(img)
    dog_labels.append(1)

# 3. Convertendo de lista para array
cat_images = np.asarray(cat_images); dog_images = np.asarray(dog_images)
cat_labels = np.asarray(cat_labels); dog_labels = np.asarray(dog_labels)

# 4. Dividindo dados para treino e teste
p = 0.7 # Porcentagem para Treino.
q = 0.1 # Porcentagem para Validação.

print('\ntrain...\n')
images_train = np.concatenate((cat_images[:int(p*cat_images.shape[0])],
                              dog_images[:int(p*dog_images.shape[0])]))
labels_train = np.concatenate((cat_labels[:int(p*cat_labels.shape[0])],
                              dog_labels[:int(p*dog_labels.shape[0])]))
print('\nval...\n')                                         
images_val = np.concatenate((cat_images[int(p*cat_images.shape[0]):int((p+q)*cat_images.shape[0])],
                              dog_images[int(p*dog_images.shape[0]):int((p+q)*dog_images.shape[0])]))
labels_val = np.concatenate((cat_labels[int(p*cat_labels.shape[0]):int((p+q)*cat_labels.shape[0])],
                              dog_labels[int(p*dog_labels.shape[0]):int((p+q)*dog_labels.shape[0])]))
print('\ntest...\n')
images_test = np.concatenate((cat_images[int((p+q)*cat_images.shape[0]):],
                              dog_images[int((p+q)*dog_images.shape[0]):]))
labels_test = np.concatenate((cat_labels[int((p+q)*cat_labels.shape[0]):],
                              dog_labels[int((p+q)*dog_labels.shape[0]):]))

# OBS: NÃO SE PREOCUPE EM SHUFFLE AGORA. ELE SERÁ DADO NO DATAGEN AO CARREGAR
# A BASE PARA O TREINAMENTO.

# 5. Saving data
print('Saving all files')
print('It will take some minutes...')
np.savez_compressed(dir_save + '/images_train', images_train) 
np.save(dir_save + '/labels_train.npy', labels_train)

np.savez_compressed(dir_save + '/images_val', images_val) 
np.save(dir_save + '/labels_val.npy', labels_val)

np.savez_compressed(dir_save + '/images_test', images_test) 
np.save(dir_save + '/labels_test.npy', labels_test)

print('Salvamento concluído!')
