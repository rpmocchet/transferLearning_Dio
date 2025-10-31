# Models Directory

Este diretório armazena os modelos treinados.

## Modelos Salvos

Após executar o treinamento, o modelo será salvo neste diretório com o nome:
- `cats_vs_dogs_model.h5` - Modelo completo em formato HDF5

## Estrutura do Modelo

O modelo utiliza **Transfer Learning** com as seguintes características:

### Arquitetura
1. **Base**: MobileNetV2 (pré-treinado no ImageNet)
   - Camadas convolucionais congeladas
   - Pesos pré-treinados preservados
   
2. **Camadas Customizadas**:
   - Global Average Pooling 2D
   - Dropout (0.2)
   - Dense (1 unidade, ativação sigmoid)

### Especificações
- **Input Shape**: (160, 160, 3)
- **Output**: Probabilidade binária (0 = gato, 1 = cachorro)
- **Parâmetros Treináveis**: ~10.000
- **Parâmetros Totais**: ~2.3M

## Carregando o Modelo

Para carregar o modelo treinado em seus próprios scripts:

```python
from tensorflow import keras

model = keras.models.load_model('models/cats_vs_dogs_model.h5')
```

## Nota

Arquivos `.h5` estão no `.gitignore` para evitar commitar modelos grandes ao repositório. Você precisará treinar o modelo localmente.
