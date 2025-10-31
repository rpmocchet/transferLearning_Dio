# Transfer Learning - Cats vs Dogs 🐱🐶

Projeto de Transfer Learning para classificação de imagens de gatos e cachorros usando TensorFlow e o dataset cats_vs_dogs.

## 📋 Descrição

Este projeto demonstra a aplicação de Transfer Learning usando uma rede neural convolucional pré-treinada (MobileNetV2) para classificar imagens de gatos e cachorros. O projeto foi desenvolvido para ser executado no Google Colab.

## 🚀 Como Usar

### Opção 1: Google Colab (Recomendado)

1. Abra o notebook no Google Colab:
   - Acesse [Google Colab](https://colab.research.google.com/)
   - Faça upload do arquivo `transfer_learning_cats_vs_dogs.ipynb`
   - Ou use: `File` → `Upload notebook` → Selecione o arquivo

2. Execute as células sequencialmente (Shift + Enter)

3. Certifique-se de ativar GPU para treinamento mais rápido:
   - `Runtime` → `Change runtime type` → `Hardware accelerator` → `GPU`

### Opção 2: Ambiente Local

```bash
# Instalar dependências
pip install tensorflow tensorflow-datasets matplotlib jupyter

# Executar Jupyter Notebook
jupyter notebook transfer_learning_cats_vs_dogs.ipynb
```

## 📊 Dataset

- **Nome**: cats_vs_dogs
- **Fonte**: TensorFlow Datasets
- **Total de imagens**: ~23,000 imagens
- **Classes**: Cat (Gato) e Dog (Cachorro)
- **Split**: 80% treino / 20% validação

## 🧠 Modelo

- **Arquitetura Base**: MobileNetV2 (pré-treinado no ImageNet)
- **Técnica**: Transfer Learning com Fine-Tuning opcional
- **Input Size**: 160x160x3
- **Camadas Customizadas**:
  - GlobalAveragePooling2D
  - Dense (128 neurônios, ReLU)
  - Dropout (0.2)
  - Dense (1 neurônio, Sigmoid) - Classificação binária

## 📈 Características do Projeto

✅ Carregamento automático do dataset via TensorFlow Datasets  
✅ Pré-processamento e normalização de imagens  
✅ Transfer Learning com modelo pré-treinado  
✅ Visualização de dados e resultados  
✅ Callbacks para Early Stopping e Learning Rate Reduction  
✅ Fine-Tuning opcional para melhor performance  
✅ Salvamento do modelo treinado  
✅ Documentação completa em português  

## 🎯 Resultados Esperados

Com Transfer Learning, é possível alcançar:
- Acurácia de validação: ~95-98%
- Treinamento rápido: 10-15 minutos (com GPU)
- Excelente generalização com poucos dados

## 📚 Conceitos Demonstrados

- Transfer Learning
- Fine-Tuning
- Data Augmentation (implícito no shuffle)
- Callbacks (EarlyStopping, ReduceLROnPlateau)
- Visualização de métricas
- Salvamento e carregamento de modelos

## 🛠️ Tecnologias Utilizadas

- **TensorFlow 2.x**: Framework de Deep Learning
- **TensorFlow Datasets**: Dataset cats_vs_dogs
- **Keras**: API de alto nível para construção de modelos
- **MobileNetV2**: Modelo base pré-treinado
- **Matplotlib**: Visualização de dados
- **NumPy**: Operações numéricas

## 📖 Estrutura do Notebook

1. Instalação e Importação de Bibliotecas
2. Carregamento e Preparação dos Dados
3. Visualização de Exemplos do Dataset
4. Construção do Modelo com Transfer Learning
5. Treinamento do Modelo
6. Avaliação e Visualização dos Resultados
7. Fine-Tuning (Opcional)
8. Salvamento do Modelo
9. Conclusão e Próximos Passos

## 🎓 Projeto DIO.me

Este projeto faz parte do curso de Transfer Learning da [Digital Innovation One (DIO)](https://www.dio.me/).

## 📝 Licença

Este projeto é open source e está disponível para fins educacionais.

## 🤝 Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir melhorias
- Adicionar novos exemplos
- Experimentar com outros modelos

## 📧 Contato

Para dúvidas ou sugestões, abra uma issue neste repositório.
