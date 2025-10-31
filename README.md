# Transfer Learning - Classificação de Gatos vs Cachorros

Projeto de Transfer Learning utilizando redes neurais convolucionais pré-treinadas para classificar imagens de gatos e cachorros. Desenvolvido como parte do curso da DIO.me.

## 📋 Descrição

Este projeto demonstra a técnica de **Transfer Learning** aplicada à classificação de imagens. Utilizamos modelos pré-treinados do TensorFlow/Keras (como MobileNetV2) treinados no dataset ImageNet e os adaptamos para classificar imagens de gatos e cachorros do dataset `cats_vs_dogs` do TensorFlow Datasets.

### O que é Transfer Learning?

Transfer Learning é uma técnica de aprendizado de máquina onde um modelo desenvolvido para uma tarefa é reutilizado como ponto de partida para um modelo em uma segunda tarefa relacionada. Isso permite:

- ✅ Treinar modelos com menos dados
- ✅ Reduzir significativamente o tempo de treinamento
- ✅ Obter melhor performance em tarefas específicas
- ✅ Aproveitar conhecimento de modelos treinados em grandes datasets

## 🚀 Começando

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/rpmocchet/transferLearning_Dio.git
cd transferLearning_Dio
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Uso

### Treinamento do Modelo

Execute o script principal para treinar o modelo:

```bash
python src/train_model.py
```

O script irá:
1. Baixar automaticamente o dataset cats_vs_dogs do TensorFlow
2. Preparar os dados (normalização, data augmentation)
3. Carregar o modelo base pré-treinado (MobileNetV2)
4. Adicionar camadas customizadas para a tarefa de classificação
5. Treinar o modelo
6. Salvar o modelo treinado em `models/`
7. Exibir métricas de performance

### Inferência com o Modelo

Para fazer predições com o modelo treinado:

```bash
python src/predict.py --image caminho/para/imagem.jpg
```

### Jupyter Notebook

Para uma exploração interativa do projeto:

```bash
jupyter notebook notebooks/transfer_learning_demo.ipynb
```

## 📁 Estrutura do Projeto

```
transferLearning_Dio/
│
├── src/                          # Código fonte
│   ├── train_model.py           # Script principal de treinamento
│   ├── predict.py               # Script de inferência
│   └── utils.py                 # Funções auxiliares
│
├── notebooks/                    # Jupyter notebooks
│   └── transfer_learning_demo.ipynb
│
├── models/                       # Modelos treinados (salvos após treinamento)
│   └── README.md
│
├── data/                         # Dados (baixados automaticamente)
│   └── README.md
│
├── docs/                         # Documentação adicional
│
├── requirements.txt              # Dependências do projeto
├── .gitignore                   # Arquivos ignorados pelo git
└── README.md                    # Este arquivo
```

## 🧠 Arquitetura do Modelo

O projeto utiliza a seguinte arquitetura:

1. **Modelo Base**: MobileNetV2 pré-treinado no ImageNet (congelado)
2. **Global Average Pooling**: Reduz dimensionalidade
3. **Dropout**: Previne overfitting
4. **Camada Densa**: Classificação binária (gato vs cachorro)

### Hiperparâmetros Principais

- **Tamanho da Imagem**: 160x160 pixels
- **Batch Size**: 32
- **Épocas**: 10
- **Learning Rate**: 0.0001
- **Otimizador**: Adam
- **Loss Function**: Binary Crossentropy

## 📊 Resultados

Após o treinamento, você poderá visualizar:

- Curvas de acurácia (treino vs validação)
- Curvas de perda (treino vs validação)
- Matriz de confusão
- Exemplos de predições

Acurácia esperada: **> 95%** no conjunto de validação.

## 🛠️ Tecnologias Utilizadas

- **TensorFlow/Keras**: Framework de deep learning
- **TensorFlow Datasets**: Dataset cats_vs_dogs
- **NumPy**: Computação numérica
- **Matplotlib**: Visualização de dados
- **Pillow**: Manipulação de imagens
- **Jupyter**: Notebooks interativos

## 📖 Conceitos Abordados

- Transfer Learning
- Redes Neurais Convolucionais (CNNs)
- Fine-tuning de modelos
- Data Augmentation
- Regularização (Dropout)
- Classificação binária de imagens

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um Fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/MinhaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abrir um Pull Request

## 📝 Licença

Este projeto é de código aberto e está disponível para fins educacionais.

## 👤 Autor

**rpmocchet**

- GitHub: [@rpmocchet](https://github.com/rpmocchet)

## 🙏 Agradecimentos

- [DIO.me](https://dio.me) - Plataforma de educação em tecnologia
- TensorFlow Team - Por disponibilizar modelos e datasets
- Comunidade Open Source

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!
