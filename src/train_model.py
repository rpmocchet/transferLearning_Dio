"""
Script principal para treinar o modelo de Transfer Learning
para classificação de gatos vs cachorros usando TensorFlow.
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Configurações
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS = 10
AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess_data():
    """
    Carrega e preprocessa o dataset cats_vs_dogs do TensorFlow Datasets.
    
    Returns:
        train_dataset: Dataset de treino processado
        validation_dataset: Dataset de validação processado
        test_dataset: Dataset de teste processado
        info: Informações sobre o dataset
    """
    print("📥 Carregando dataset cats_vs_dogs...")
    
    # Carregar dataset
    (train_ds, validation_ds, test_ds), info = tfds.load(
        'cats_vs_dogs',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    
    num_classes = info.features['label'].num_classes
    print(f"✅ Dataset carregado com sucesso!")
    print(f"   - Número de classes: {num_classes}")
    print(f"   - Total de imagens: {info.splits['train'].num_examples}")
    
    return train_ds, validation_ds, test_ds, info


def preprocess_image(image, label):
    """
    Preprocessa uma imagem: redimensiona e normaliza.
    
    Args:
        image: Imagem a ser processada
        label: Label da imagem
        
    Returns:
        image: Imagem processada
        label: Label
    """
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def augment_image(image, label):
    """
    Aplica data augmentation para aumentar a diversidade do dataset.
    
    Args:
        image: Imagem a ser aumentada
        label: Label da imagem
        
    Returns:
        image: Imagem aumentada
        label: Label
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    return image, label


def prepare_dataset(ds, augment=False, shuffle=False, cache=True):
    """
    Prepara o dataset para treinamento aplicando transformações.
    
    Args:
        ds: Dataset do TensorFlow
        augment: Se True, aplica data augmentation
        shuffle: Se True, embaralha os dados
        cache: Se True, mantém dados em cache
        
    Returns:
        Dataset processado
    """
    if cache:
        ds = ds.cache()
    
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    
    ds = ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds


def create_model():
    """
    Cria o modelo de Transfer Learning usando MobileNetV2 pré-treinado.
    
    Returns:
        model: Modelo compilado pronto para treinamento
    """
    print("\n🏗️  Construindo modelo de Transfer Learning...")
    
    # Carregar modelo base pré-treinado (sem as camadas de classificação)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar as camadas do modelo base
    base_model.trainable = False
    
    # Criar camadas de classificação customizadas
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compilar o modelo
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Modelo criado com sucesso!")
    print(f"\n📊 Resumo do modelo:")
    model.summary()
    
    return model


def plot_training_history(history):
    """
    Plota gráficos do histórico de treinamento.
    
    Args:
        history: Objeto History retornado pelo fit()
    """
    print("\n📈 Gerando gráficos de treinamento...")
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    
    # Gráfico de acurácia
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Acurácia de Treino')
    plt.plot(epochs_range, val_acc, label='Acurácia de Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia de Treino e Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    
    # Gráfico de perda
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Perda de Treino')
    plt.plot(epochs_range, val_loss, label='Perda de Validação')
    plt.legend(loc='upper right')
    plt.title('Perda de Treino e Validação')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    
    plt.tight_layout()
    
    # Salvar gráfico
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'models/training_history_{timestamp}.png')
    print(f"✅ Gráficos salvos em models/training_history_{timestamp}.png")
    
    plt.show()


def evaluate_model(model, test_dataset):
    """
    Avalia o modelo no dataset de teste.
    
    Args:
        model: Modelo treinado
        test_dataset: Dataset de teste
    """
    print("\n🎯 Avaliando modelo no conjunto de teste...")
    
    loss, accuracy = model.evaluate(test_dataset)
    
    print(f"\n📊 Resultados no conjunto de teste:")
    print(f"   - Perda: {loss:.4f}")
    print(f"   - Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")


def main():
    """
    Função principal que executa todo o pipeline de treinamento.
    """
    print("=" * 60)
    print("🐱 🐶 TRANSFER LEARNING - CATS VS DOGS 🐶 🐱")
    print("=" * 60)
    
    # 1. Carregar e preparar dados
    train_ds, val_ds, test_ds, info = load_and_preprocess_data()
    
    print("\n⚙️  Preparando datasets...")
    train_dataset = prepare_dataset(train_ds, augment=True, shuffle=True)
    validation_dataset = prepare_dataset(val_ds)
    test_dataset = prepare_dataset(test_ds)
    print("✅ Datasets preparados!")
    
    # 2. Criar modelo
    model = create_model()
    
    # 3. Treinar modelo
    print(f"\n🚀 Iniciando treinamento ({EPOCHS} épocas)...")
    print("-" * 60)
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=validation_dataset,
        verbose=1
    )
    
    print("-" * 60)
    print("✅ Treinamento concluído!")
    
    # 4. Avaliar modelo
    evaluate_model(model, test_dataset)
    
    # 5. Salvar modelo
    print("\n💾 Salvando modelo...")
    os.makedirs('models', exist_ok=True)
    model_path = 'models/cats_vs_dogs_model.h5'
    model.save(model_path)
    print(f"✅ Modelo salvo em: {model_path}")
    
    # 6. Plotar histórico
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("🎉 Processo concluído com sucesso!")
    print("=" * 60)
    print("\n💡 Próximos passos:")
    print("   1. Use 'python src/predict.py --image caminho/imagem.jpg' para fazer predições")
    print("   2. Explore o notebook em 'notebooks/transfer_learning_demo.ipynb'")
    print("   3. Experimente fine-tuning descongelando algumas camadas do modelo base")


if __name__ == "__main__":
    main()
