"""
Script para fazer predições com o modelo treinado.
"""

import tensorflow as tf
import numpy as np
import argparse
import os
from PIL import Image
import matplotlib.pyplot as plt


IMG_SIZE = 160
CLASS_NAMES = ['Gato', 'Cachorro']


def load_model(model_path='models/cats_vs_dogs_model.h5'):
    """
    Carrega o modelo treinado.
    
    Args:
        model_path: Caminho para o arquivo do modelo
        
    Returns:
        model: Modelo carregado
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ Modelo não encontrado em '{model_path}'.\n"
            "   Execute primeiro: python src/train_model.py"
        )
    
    print(f"📥 Carregando modelo de {model_path}...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Modelo carregado com sucesso!")
    
    return model


def preprocess_image(image_path):
    """
    Carrega e preprocessa uma imagem para predição.
    
    Args:
        image_path: Caminho para a imagem
        
    Returns:
        image: Imagem preprocessada
        original_image: Imagem original para visualização
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Imagem não encontrada: {image_path}")
    
    # Carregar imagem original
    original_image = Image.open(image_path)
    
    # Converter para RGB se necessário
    if original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')
    
    # Preprocessar para o modelo
    image = original_image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = image.astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)  # Adicionar dimensão de batch
    
    return image, original_image


def predict(model, image_path, show_plot=True):
    """
    Faz predição para uma imagem.
    
    Args:
        model: Modelo treinado
        image_path: Caminho para a imagem
        show_plot: Se True, mostra a imagem com a predição
        
    Returns:
        prediction: Probabilidade predita
        class_name: Nome da classe predita
    """
    print(f"\n🔍 Analisando imagem: {image_path}")
    
    # Preprocessar imagem
    image, original_image = preprocess_image(image_path)
    
    # Fazer predição
    prediction = model.predict(image, verbose=0)[0][0]
    
    # Interpretar resultado
    if prediction > 0.5:
        class_idx = 1
        confidence = prediction
    else:
        class_idx = 0
        confidence = 1 - prediction
    
    class_name = CLASS_NAMES[class_idx]
    
    # Mostrar resultado
    print(f"\n📊 Resultado da predição:")
    print(f"   - Classe: {class_name}")
    print(f"   - Confiança: {confidence*100:.2f}%")
    print(f"   - Score bruto: {prediction:.4f}")
    
    # Visualizar
    if show_plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(original_image)
        plt.axis('off')
        plt.title(f'Predição: {class_name} ({confidence*100:.1f}% de confiança)', 
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return prediction, class_name


def predict_batch(model, image_paths):
    """
    Faz predições para múltiplas imagens.
    
    Args:
        model: Modelo treinado
        image_paths: Lista de caminhos para as imagens
    """
    print(f"\n🔍 Analisando {len(image_paths)} imagens...")
    
    results = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}]", end=" ")
        try:
            prediction, class_name = predict(model, image_path, show_plot=False)
            results.append({
                'path': image_path,
                'class': class_name,
                'prediction': prediction
            })
        except Exception as e:
            print(f"❌ Erro ao processar {image_path}: {str(e)}")
    
    # Resumo
    print("\n" + "="*60)
    print("📊 RESUMO DAS PREDIÇÕES")
    print("="*60)
    
    gatos = sum(1 for r in results if r['class'] == 'Gato')
    cachorros = sum(1 for r in results if r['class'] == 'Cachorro')
    
    print(f"\nTotal de imagens processadas: {len(results)}")
    print(f"   - Gatos: {gatos}")
    print(f"   - Cachorros: {cachorros}")
    
    print("\nDetalhes:")
    for result in results:
        filename = os.path.basename(result['path'])
        confidence = result['prediction'] if result['class'] == 'Cachorro' else 1 - result['prediction']
        print(f"   {filename:30s} -> {result['class']:10s} ({confidence*100:.1f}%)")


def main():
    """
    Função principal.
    """
    parser = argparse.ArgumentParser(
        description='Fazer predições com o modelo de classificação de gatos vs cachorros',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python src/predict.py --image data/cat.jpg
  python src/predict.py --image data/dog.jpg --model models/cats_vs_dogs_model.h5
  python src/predict.py --images data/img1.jpg data/img2.jpg data/img3.jpg
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Caminho para uma única imagem'
    )
    
    parser.add_argument(
        '--images',
        type=str,
        nargs='+',
        help='Caminhos para múltiplas imagens'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='models/cats_vs_dogs_model.h5',
        help='Caminho para o modelo (padrão: models/cats_vs_dogs_model.h5)'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.image and not args.images:
        parser.error("❌ Você deve fornecer pelo menos --image ou --images")
    
    print("=" * 60)
    print("🐱 🐶 PREDIÇÃO - CATS VS DOGS 🐶 🐱")
    print("=" * 60)
    
    try:
        # Carregar modelo
        model = load_model(args.model)
        
        # Fazer predições
        if args.image:
            predict(model, args.image)
        elif args.images:
            predict_batch(model, args.images)
        
        print("\n" + "=" * 60)
        print("✅ Predições concluídas!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
