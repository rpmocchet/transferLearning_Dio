# Data Directory

Este diretório armazena os dados utilizados no projeto.

## Dataset: cats_vs_dogs

O dataset será baixado automaticamente pelo TensorFlow Datasets quando você executar o script de treinamento pela primeira vez.

### Informações sobre o Dataset

- **Nome**: cats_vs_dogs
- **Fonte**: TensorFlow Datasets
- **Tamanho**: ~800MB
- **Imagens de Treino**: ~23.000 imagens
- **Classes**: 2 (gatos e cachorros)

### Estrutura

Após o download, os dados serão organizados automaticamente pelo TensorFlow Datasets em cache local. Você não precisa organizar manualmente as imagens.

## Dados Personalizados

Se você quiser testar o modelo com suas próprias imagens:

1. Adicione suas imagens neste diretório
2. Use o script `src/predict.py` para fazer predições:

```bash
python src/predict.py --image data/minha_imagem.jpg
```

## Nota

Este diretório está no `.gitignore` para evitar commitar grandes arquivos de dados ao repositório.
