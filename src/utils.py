"""
Funções auxiliares para o projeto de Transfer Learning.
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List


def visualize_dataset_samples(dataset, class_names: List[str], num_samples: int = 9):
    """
    Visualiza amostras aleatórias do dataset.
    
    Args:
        dataset: TensorFlow dataset
        class_names: Lista com nomes das classes
        num_samples: Número de amostras a visualizar
    """
    plt.figure(figsize=(12, 12))
    
    for images, labels in dataset.take(1):
        for i in range(min(num_samples, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            
            label = labels[i].numpy()
            class_name = class_names[int(label)]
            
            plt.title(f'{class_name}')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names: List[str]):
    """
    Plota matriz de confusão.
    
    Args:
        y_true: Labels verdadeiros
        y_pred: Labels preditos
        class_names: Lista com nomes das classes
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Calcular matriz de confusão
    cm = confusion_matrix(y_true, y_pred)
    
    # Plotar
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusão')
    plt.ylabel('Classe Verdadeira')
    plt.xlabel('Classe Predita')
    plt.tight_layout()
    plt.show()


def get_predictions_from_dataset(model, dataset, steps: int = None):
    """
    Obtém predições e labels verdadeiros de um dataset.
    
    Args:
        model: Modelo treinado
        dataset: TensorFlow dataset
        steps: Número de batches a processar
        
    Returns:
        y_true: Array com labels verdadeiros
        y_pred: Array com predições
        y_pred_proba: Array com probabilidades das predições
    """
    y_true = []
    y_pred_proba = []
    
    for i, (images, labels) in enumerate(dataset):
        if steps and i >= steps:
            break
            
        predictions = model.predict(images, verbose=0)
        
        y_true.extend(labels.numpy())
        y_pred_proba.extend(predictions.flatten())
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    return y_true, y_pred, y_pred_proba


def plot_sample_predictions(model, dataset, class_names: List[str], num_samples: int = 9):
    """
    Visualiza predições do modelo em amostras aleatórias.
    
    Args:
        model: Modelo treinado
        dataset: TensorFlow dataset
        class_names: Lista com nomes das classes
        num_samples: Número de amostras a visualizar
    """
    plt.figure(figsize=(15, 15))
    
    for images, labels in dataset.take(1):
        predictions = model.predict(images, verbose=0)
        
        for i in range(min(num_samples, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            
            true_label = int(labels[i].numpy())
            pred_label = int(predictions[i] > 0.5)
            confidence = predictions[i][0] if pred_label == 1 else 1 - predictions[i][0]
            
            true_class = class_names[true_label]
            pred_class = class_names[pred_label]
            
            color = 'green' if true_label == pred_label else 'red'
            
            plt.title(f'Real: {true_class}\nPred: {pred_class} ({confidence*100:.1f}%)',
                     color=color, fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def create_learning_rate_scheduler(initial_lr: float = 0.0001, 
                                   decay_steps: int = 100,
                                   decay_rate: float = 0.96):
    """
    Cria um scheduler de learning rate exponencial.
    
    Args:
        initial_lr: Learning rate inicial
        decay_steps: Número de steps para decaimento
        decay_rate: Taxa de decaimento
        
    Returns:
        Learning rate scheduler
    """
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    
    return lr_schedule


def save_model_summary(model, filepath: str = 'models/model_summary.txt'):
    """
    Salva o resumo do modelo em arquivo de texto.
    
    Args:
        model: Modelo do Keras
        filepath: Caminho para salvar o arquivo
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"✅ Resumo do modelo salvo em: {filepath}")


def calculate_model_size(model):
    """
    Calcula o tamanho do modelo em MB.
    
    Args:
        model: Modelo do Keras
        
    Returns:
        Tamanho do modelo em MB
    """
    # Contar parâmetros
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    # Assumindo float32 (4 bytes por parâmetro)
    size_mb = (total_params * 4) / (1024 * 1024)
    
    print(f"📊 Informações do modelo:")
    print(f"   - Parâmetros treináveis: {trainable_params:,}")
    print(f"   - Parâmetros não treináveis: {non_trainable_params:,}")
    print(f"   - Total de parâmetros: {total_params:,}")
    print(f"   - Tamanho estimado: {size_mb:.2f} MB")
    
    return size_mb


def print_dataset_info(dataset, name: str = "Dataset"):
    """
    Imprime informações sobre um dataset.
    
    Args:
        dataset: TensorFlow dataset
        name: Nome do dataset
    """
    print(f"\n📊 Informações sobre {name}:")
    
    # Obter uma amostra
    for images, labels in dataset.take(1):
        batch_size = len(images)
        img_shape = images[0].shape
        
        print(f"   - Batch size: {batch_size}")
        print(f"   - Shape da imagem: {img_shape}")
        print(f"   - Dtype da imagem: {images.dtype}")
        print(f"   - Range de valores: [{images.numpy().min():.2f}, {images.numpy().max():.2f}]")
        print(f"   - Labels: {labels.numpy()[:min(5, batch_size)]}")
