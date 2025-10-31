# Guia de Contribuição

Obrigado por considerar contribuir para este projeto! 🎉

## 🤝 Como Contribuir

### Reportar Bugs

Se você encontrar um bug, por favor abra uma [issue](https://github.com/rpmocchet/transferLearning_Dio/issues) incluindo:

- Descrição clara do problema
- Passos para reproduzir
- Comportamento esperado vs comportamento atual
- Versão do Python e TensorFlow
- Sistema operacional

### Sugerir Melhorias

Tem uma ideia para melhorar o projeto? Ótimo! Abra uma issue com:

- Descrição da melhoria
- Por que seria útil
- Como você imagina que funcionaria

### Contribuir com Código

1. **Fork o repositório**
   ```bash
   # Clique em "Fork" no GitHub
   ```

2. **Clone seu fork**
   ```bash
   git clone https://github.com/seu-usuario/transferLearning_Dio.git
   cd transferLearning_Dio
   ```

3. **Crie uma branch para sua feature**
   ```bash
   git checkout -b feature/minha-feature
   ```

4. **Faça suas mudanças**
   - Escreva código claro e bem documentado
   - Adicione comentários em português
   - Siga o estilo do código existente

5. **Teste suas mudanças**
   ```bash
   python verify_setup.py
   python src/train_model.py  # Se aplicável
   ```

6. **Commit suas mudanças**
   ```bash
   git add .
   git commit -m "Adiciona: descrição clara da mudança"
   ```

7. **Push para seu fork**
   ```bash
   git push origin feature/minha-feature
   ```

8. **Abra um Pull Request**
   - Vá para o repositório original no GitHub
   - Clique em "New Pull Request"
   - Descreva suas mudanças claramente

## 📝 Diretrizes de Código

### Estilo Python

- Siga PEP 8
- Use nomes descritivos em inglês para variáveis e funções
- Comentários e docstrings em português
- Máximo 100 caracteres por linha

### Exemplo de Docstring

```python
def minha_funcao(parametro1, parametro2):
    """
    Descrição breve da função.
    
    Args:
        parametro1: Descrição do primeiro parâmetro
        parametro2: Descrição do segundo parâmetro
        
    Returns:
        Descrição do retorno
    """
    pass
```

### Commits

Use mensagens de commit claras:

- ✅ "Adiciona função para carregar dados customizados"
- ✅ "Corrige bug no preprocessamento de imagens"
- ✅ "Atualiza documentação sobre fine-tuning"
- ❌ "fix"
- ❌ "update"
- ❌ "changes"

## 🎯 Áreas para Contribuição

Algumas ideias de onde você pode ajudar:

### Código
- [ ] Suporte para novos modelos base (ResNet, EfficientNet, etc.)
- [ ] Implementação de callbacks customizados
- [ ] Scripts para fine-tuning avançado
- [ ] Interface web com Streamlit/Flask
- [ ] Suporte para múltiplas classes (além de binário)
- [ ] Exportação para TensorFlow Lite

### Documentação
- [ ] Tutoriais em vídeo
- [ ] Exemplos de uso avançado
- [ ] Tradução para outros idiomas
- [ ] FAQ (Perguntas Frequentes)
- [ ] Guia de troubleshooting

### Testes
- [ ] Testes unitários
- [ ] Testes de integração
- [ ] Validação de modelos

### Datasets
- [ ] Exemplos com outros datasets
- [ ] Scripts para criar datasets customizados
- [ ] Data augmentation avançado

## 🐛 Reportando Problemas

Antes de reportar um problema:

1. Verifique se já não existe uma issue similar
2. Certifique-se de estar usando a versão mais recente
3. Tente reproduzir em ambiente limpo (venv novo)

## 💬 Processo de Review

Após abrir um PR:

1. Mantenedor irá revisar em até 7 dias
2. Pode haver pedidos de mudanças
3. Após aprovação, será feito merge
4. Seu nome será adicionado aos contribuidores!

## 🏆 Reconhecimento

Todos os contribuidores serão:

- Listados no README
- Mencionados nas release notes
- Eternamente agradecidos! 🙏

## 📜 Código de Conduta

### Nossos Compromissos

Este projeto se compromete a fornecer uma experiência livre de assédio para todos.

### Comportamentos Esperados

- Usar linguagem acolhedora e inclusiva
- Respeitar pontos de vista diferentes
- Aceitar críticas construtivas
- Focar no melhor para a comunidade

### Comportamentos Inaceitáveis

- Uso de linguagem ou imagens sexualizadas
- Comentários insultuosos ou depreciativos
- Assédio público ou privado
- Publicar informações privadas de outros

## 🤔 Dúvidas?

Se tiver dúvidas sobre como contribuir:

1. Leia a [documentação](docs/GUIDE.md)
2. Procure em [Issues](https://github.com/rpmocchet/transferLearning_Dio/issues)
3. Abra uma issue com sua dúvida

## 📞 Contato

- GitHub: [@rpmocchet](https://github.com/rpmocchet)
- Issues: [GitHub Issues](https://github.com/rpmocchet/transferLearning_Dio/issues)

---

Obrigado por contribuir! Juntos fazemos este projeto melhor! 🚀
