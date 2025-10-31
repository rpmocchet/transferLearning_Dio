# Fluxo de Trabalho do Projeto

Este documento descreve o fluxo completo de trabalho para usar este projeto de Transfer Learning.

## 🎯 Objetivo

Criar um classificador de imagens capaz de distinguir entre gatos e cachorros usando Transfer Learning com TensorFlow.

## 📋 Pré-requisitos

Antes de começar, certifique-se de ter:

- Python 3.8 ou superior instalado
- pip (gerenciador de pacotes Python)
- 2GB de espaço em disco livre (para dataset e modelos)
- Conexão com internet (para download do dataset)

## 🚀 Passos para Começar

### 1. Configuração Inicial

```bash
# Clone o repositório
git clone https://github.com/rpmocchet/transferLearning_Dio.git
cd transferLearning_Dio

# Crie um ambiente virtual (recomendado)
python -m venv venv

# Ative o ambiente virtual
# No Linux/Mac:
source venv/bin/activate
# No Windows:
venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### 2. Verificar Instalação

```bash
# Execute o script de verificação
python verify_setup.py
```

Se tudo estiver correto, você verá: ✅ VERIFICAÇÃO CONCLUÍDA COM SUCESSO!

### 3. Primeiro Treinamento

```bash
# Execute o script de treinamento
python src/train_model.py
```

**O que acontecerá:**

1. Download do dataset cats_vs_dogs (~800MB) - primeira vez apenas
2. Preprocessamento das imagens
3. Carregamento do modelo MobileNetV2 pré-treinado
4. Treinamento por 10 épocas (~10-30 minutos dependendo do hardware)
5. Avaliação no conjunto de teste
6. Salvamento do modelo em `models/cats_vs_dogs_model.h5`
7. Geração de gráficos de treinamento

**Saída esperada:**
```
🐱 🐶 TRANSFER LEARNING - CATS VS DOGS 🐶 🐱
📥 Carregando dataset cats_vs_dogs...
✅ Dataset carregado com sucesso!
🏗️  Construindo modelo de Transfer Learning...
🚀 Iniciando treinamento (10 épocas)...
Epoch 1/10
...
✅ Treinamento concluído!
🎯 Avaliando modelo no conjunto de teste...
📊 Resultados no conjunto de teste:
   - Acurácia: 0.9565 (95.65%)
💾 Modelo salvo em: models/cats_vs_dogs_model.h5
```

### 4. Fazer Predições

Após o treinamento, use o modelo para classificar novas imagens:

```bash
# Predição em uma única imagem
python src/predict.py --image caminho/para/imagem.jpg

# Predições em múltiplas imagens
python src/predict.py --images img1.jpg img2.jpg img3.jpg
```

**Exemplo de saída:**
```
🐱 🐶 PREDIÇÃO - CATS VS DOGS 🐶 🐱
📥 Carregando modelo de models/cats_vs_dogs_model.h5...
✅ Modelo carregado com sucesso!

🔍 Analisando imagem: cat_photo.jpg

📊 Resultado da predição:
   - Classe: Gato
   - Confiança: 98.75%
   - Score bruto: 0.0125
```

### 5. Exploração Interativa

Para uma experiência mais interativa:

```bash
# Inicie o Jupyter Notebook
jupyter notebook notebooks/transfer_learning_demo.ipynb
```

O notebook permite:
- Visualizar amostras do dataset
- Entender o preprocessamento
- Treinar o modelo passo a passo
- Visualizar predições
- Experimentar com fine-tuning

## 📊 Fluxo de Dados

```
Dataset TensorFlow
      ↓
Preprocessamento
(Resize, Normalização)
      ↓
Data Augmentation
(Flip, Brightness, Contrast)
      ↓
Modelo MobileNetV2
(Feature Extraction)
      ↓
Camadas Customizadas
(Pooling, Dropout, Dense)
      ↓
Predição
(Gato ou Cachorro)
```

## 🎨 Estrutura de Arquivos Após Setup

```
transferLearning_Dio/
│
├── data/
│   └── tensorflow_datasets/        # Dataset baixado (gerado)
│
├── models/
│   ├── cats_vs_dogs_model.h5      # Modelo treinado (gerado)
│   └── training_history_*.png     # Gráficos (gerado)
│
├── src/
│   ├── __init__.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
│
├── notebooks/
│   └── transfer_learning_demo.ipynb
│
└── docs/
    ├── GUIDE.md
    └── WORKFLOW.md
```

## 🔄 Ciclo de Desenvolvimento

### Ciclo Básico

1. **Treinar** → `python src/train_model.py`
2. **Avaliar** → Verificar métricas no terminal
3. **Testar** → `python src/predict.py --image test.jpg`
4. **Iterar** → Ajustar hiperparâmetros e repetir

### Ciclo Avançado

1. **Experimentar** → Modificar hiperparâmetros
2. **Comparar** → Treinar múltiplas versões
3. **Selecionar** → Escolher melhor modelo
4. **Fine-tune** → Ajuste fino das camadas
5. **Deploy** → Usar em produção

## 🎛️ Personalizações Comuns

### Alterar Número de Épocas

Edite `src/train_model.py`:
```python
EPOCHS = 20  # Aumentar para mais treinamento
```

### Mudar Tamanho do Batch

Edite `src/train_model.py`:
```python
BATCH_SIZE = 16  # Reduzir se houver erro de memória
```

### Usar Modelo Base Diferente

Edite `src/train_model.py`:
```python
# Trocar de MobileNetV2 para ResNet50
base_model = tf.keras.applications.ResNet50(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
```

### Ajustar Learning Rate

Edite `src/train_model.py`:
```python
optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001)
```

## 🐛 Resolução de Problemas

### Erro: "Module 'tensorflow' not found"

**Solução:**
```bash
pip install tensorflow
```

### Erro: "Out of Memory"

**Solução:**
```python
# Reduzir batch size em train_model.py
BATCH_SIZE = 16  # ou menor
```

### Dataset não baixa

**Solução:**
1. Verifique conexão com internet
2. Tente novamente (pode haver timeout)
3. Verifique espaço em disco

### Acurácia muito baixa (<80%)

**Possíveis causas:**
1. Poucas épocas de treinamento
2. Learning rate inadequado
3. Problema no preprocessamento

**Solução:**
1. Aumente número de épocas
2. Experimente learning rates diferentes
3. Verifique visualização dos dados

## 📈 Monitoramento do Treinamento

Durante o treinamento, observe:

1. **Loss diminuindo**: Boa! Modelo está aprendendo
2. **Accuracy aumentando**: Ótimo! Modelo está melhorando
3. **Val_loss vs Loss**: 
   - Próximos: Modelo generaliza bem
   - Val_loss muito maior: Possível overfitting

## 🎯 Métricas de Sucesso

| Métrica | Ruim | Regular | Bom | Excelente |
|---------|------|---------|-----|-----------|
| Accuracy | <80% | 80-90% | 90-95% | >95% |
| Loss | >0.5 | 0.3-0.5 | 0.1-0.3 | <0.1 |
| Training Time | >1h | 30-60min | 15-30min | <15min |

## 🚀 Próximos Passos

Após dominar o básico:

1. **Experimente outros datasets**
   - CIFAR-10 (10 classes)
   - Fashion MNIST (roupas)
   - Seus próprios dados

2. **Explore arquiteturas diferentes**
   - ResNet50
   - EfficientNet
   - InceptionV3

3. **Implemente callbacks**
   - EarlyStopping
   - ModelCheckpoint
   - ReduceLROnPlateau

4. **Deploy do modelo**
   - TensorFlow Serving
   - TensorFlow Lite (mobile)
   - Web app com Flask/Streamlit

## 📚 Recursos de Aprendizado

### Para Iniciantes
1. Leia o [README.md](../README.md)
2. Execute o notebook passo a passo
3. Leia o [GUIDE.md](GUIDE.md) completo

### Para Intermediários
1. Modifique hiperparâmetros
2. Experimente outros modelos base
3. Implemente callbacks customizados

### Para Avançados
1. Implemente fine-tuning progressivo
2. Experimente mixed precision training
3. Otimize para deployment em produção

## 💡 Dicas Importantes

1. **Sempre use ambiente virtual** - Evita conflitos de dependências
2. **Comece com poucos epochs** - Valide que tudo funciona antes de treinar muito
3. **Monitore val_loss** - Indica se modelo generaliza bem
4. **Salve versões do modelo** - Use timestamps nos nomes
5. **Documente experimentos** - Anote o que funcionou/não funcionou

## 🤝 Contribuindo

Quer melhorar o projeto?

1. Fork o repositório
2. Crie uma branch (`git checkout -b feature/melhoria`)
3. Commit suas mudanças (`git commit -m 'Adiciona melhoria'`)
4. Push para a branch (`git push origin feature/melhoria`)
5. Abra um Pull Request

## 📞 Suporte

Problemas ou dúvidas?

1. Verifique a [documentação](GUIDE.md)
2. Procure em [Issues](https://github.com/rpmocchet/transferLearning_Dio/issues)
3. Abra uma nova issue se necessário

---

**Bom treinamento! 🚀🐱🐶**
