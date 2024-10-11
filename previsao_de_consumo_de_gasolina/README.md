# Projeto de Previsão de Consumo de Combustível

Este projeto utiliza um conjunto de dados de especificações técnicas de carros para prever o consumo de combustível em milhas por galão (MPG) utilizando redes neurais com a biblioteca Keras.

## Sumário

- [Objetivo](#objetivo)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Dataset](#dataset)
- [Etapas do Projeto](#etapas-do-projeto)
- [Como Executar](#como-executar)
- [Resultados](#resultados)
- [Conclusão](#conclusão)

## Objetivo

O objetivo deste projeto é construir um modelo preditivo que possa estimar o consumo de combustível de veículos com base em suas características técnicas.

## Tecnologias Utilizadas

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Matplotlib

## Dataset

O dataset utilizado neste projeto foi retirado do [Kaggle](https://www.kaggle.com/datasets/uciml/autompg-dataset?resource=download). O conjunto de dados contém informações sobre diferentes modelos de carros, incluindo atributos como cilindrada, potência, peso, entre outros.
### Atributos Principais

- `MPG`: Consumo em milhas por galão (target).
- `Cilindrada`: Cilindrada do motor.
- `Cilindros`: Número de cilindros.
- `Potência`: Potência do motor.
- `Peso`: Peso do veículo.
- `Aceleração`: Aceleração do veículo.
- `Ano do Modelo`: Ano de fabricação.
- `Origem`: País de origem do veículo.

## Etapas do Projeto

1. **Carregamento do Dataset**: Carregamento e pré-processamento dos dados.
2. **Tratamento de Dados**: Normalização dos atributos e separação dos rótulos.
3. **Divisão em Conjuntos de Treinamento e Teste**: Criação de conjuntos para treinamento e teste do modelo.
4. **Construção do Modelo**: Criação de uma rede neural utilizando Keras.
5. **Treinamento do Modelo**: Treinamento do modelo com os dados normalizados.
6. **Avaliação do Modelo**: Avaliação do desempenho utilizando métricas como MAE e MSE.
7. **Previsões**: Geração de previsões e comparação com os rótulos reais.
8. **Exportação do Modelo**: Exportação do modelo treinado em formatos HDF5 e TensorFlow Lite.

## Como Executar

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu_usuario/seu_repositorio.git
