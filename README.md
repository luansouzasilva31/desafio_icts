# desafio_icts
Implementação referente ao desafio proposto em "desafio_dl.pdf".

## organização
 Aqui no GitHub só será disponibilizado os algoritmos, devido à limitação de espaço. Caso sinta interesse no acesso à base de dados, confira em: https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset
 
 ### Estrutura:
 `
 |_ binary_classification
    |_ cats_and_dogs_dataset
        |_ annotations
        |_ images/images
        |_ npz (pasta criada pelo algoritmo a_preprocessingDataset.py)
    |_ pyscripts
        |_ a_preprocessingDataset.py
        |_ b_trainingModel.py
        |_ c_evaluatingModel.py
    |_ weights
        |_ model.h5 (resultante do algoritmo b_trainingModel.py)
 |_ categorical_classification
     |_ cats_and_dogs_dataset
        |_ annotations
        |_ images/images
        |_ npz (pasta criada pelo algoritmo a_preprocessingDataset.py)
    |_ pyscripts
        |_ a_preprocessingDataset.py
        |_ b_trainingModel.py
        |_ c_evaluatingModel.py
    |_ weights
        |_ model.h5 (resultante do algoritmo b_trainingModel.py)
        |_ model_last.h5 (melhor otimizado. Usado na avaliação do modelo em c_evaluatingModel.py)
 |_ desafio_dl.pdf
`
 A pasta se divide em duas partes: binary_classification, onde encontram-se os arquivos referentes à classificação binária para gato/cachorro;
 categorical_crossentropy, onde encontram-se os arquivos referentes à classificação categórica de raças (37 ao todo).

 Cada diretório possui a base de dados original, para que os trabalhos possam ser baixados isoladamente, de acordo com o interesse da pessoa.

 Tomando como exemplo a pasta binary_classification:
    todos os scripts estão na pasta /pyscripts
    todos os pesos resultantes do treinamento da rede estão na pasta /weights
    toda a base de dados estã em /cats_and_dogs_dataset
        /annotations contém os metadados do dataset (arquivos xml)
        /images/images contém o dataset, que são imagens .jpg
        /npz contém o dataset compilado no formato .npz, feito pelo código "a_preprocessingDataset.py".

## algoritmo
Em cada caso é realizado um pré-processamento, que carrega a base de acordo com as características para treinamento (binário ou categórico). No caso binário, o algoritmo divide as classes pelo nome das imagens que começam com letra maiúscula (gato) ou minúscula (cachorro). Já no caso categórico, que se centra na classificação de raças, o algoritmo se baseia no nome de cada imagem da base de dados da seguinte forma: a string que vai do início até o último underline do nome de cada imagem corresponde à raça. Exemplo: Raça_de_cachorro1_34.jpg => a raça seria "Raça_de_cachorro1". Veja que tudo após o último underline foi ignorado.

Para ambos os casos, foi utilizado a técnica de  transfer learning, que importa redes pré-treinadas incorporadas ao Keras, e então aplica um fine-tuning para adaptá-las ao problema proposto.

CLASSIFICAÇÃO BINÁRIA: foi importada a rede built-in VGG16, habilitando para fine-tuning apenas 'block5_conv1', que é um bloco convolucional da rede. 
CLASSIFICAÇÃO CATEGÓRICA: foi utilizada a rede built-in Xception.

Após essas redes, é adicionada uma parte densa para fazer as classificações binária e categórica.

## resultados
Binary_classification: 98,041% de accuracy (base de teste)
Categorical_classification: 89,315% de accuracy (base de Teste)

### Para mais detalhes, verifique o código. Cada etapa realizada está devidamente comentada.


