import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# --- PARTE 1: SIMULAÇÃO E PREPARAÇÃO DOS DADOS (CÓDIGO QUE VOCÊ JÁ TEM) ---

# Definindo as opções para cada variável
opcoes_sim_nao = ['Sim', 'Nao']
opcoes_urgencia = ['Alta', 'Media', 'Baixa']
opcoes_clientes = [f'Cliente_{i}' for i in range(1, 6)]
opcoes_etiquetas = ['Etiqueta_105mm', 'Etiqueta_64mm']

# Número de exemplos (linhas) que queremos simular para treinar o modelo
num_amostras = 200

# Gerando dados simulados para as variáveis
data_simulada_simplificada = {
    'Estoque_OK': np.random.choice(opcoes_sim_nao, num_amostras, p=[0.7, 0.3]),
    'Dias_Desde_Pedido': np.random.randint(1, 30, num_amostras),
    'Urgencia_Pedido': np.random.choice(opcoes_urgencia, num_amostras, p=[0.4, 0.3, 0.3]),
    'Nome_Cliente': np.random.choice(opcoes_clientes, num_amostras),
    'Tempo_Medio_Producao_Min': np.random.randint(20, 180, num_amostras),
}

df_simples = pd.DataFrame(data_simulada_simplificada)

df_simples['Produto_A_Ser_Produzido_Cliente'] = ''

for i in range(len(df_simples)):
    cliente_atual = df_simples.loc[i, 'Nome_Cliente']
    if df_simples.loc[i, 'Estoque_OK'] == 'Sim' and df_simples.loc[i, 'Urgencia_Pedido'] == 'Alta':
        produto_escolhido = np.random.choice(['Etiqueta_105mm', 'Etiqueta_64mm'], p=[0.7, 0.3])
    else:
        produto_escolhido = np.random.choice(opcoes_etiquetas)
    df_simples.loc[i, 'Produto_A_Ser_Produzido_Cliente'] = f"{cliente_atual}_{produto_escolhido}"

# Separando X e y e aplicando One-Hot Encoding
X_simples = df_simples.drop('Produto_A_Ser_Produzido_Cliente', axis=1)
y_simples = df_simples['Produto_A_Ser_Produzido_Cliente']

X_simples = pd.get_dummies(X_simples, columns=['Estoque_OK', 'Urgencia_Pedido', 'Nome_Cliente'])

# Dividindo dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_simples, y_simples, test_size=0.20, random_state=42)

# Treinando o modelo (aqui ele é treinado uma vez quando o app inicia)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Armazenar as colunas de treinamento para garantir consistência nas previsões
colunas_treinamento = X_train.columns

# --- FIM DA PARTE 1 ---

# --- PARTE 2: INTERFACE COM STREAMLIT ---

st.title('Sistema de Sugestão de Produção de Etiquetas')
st.write('Use este painel para obter a sugestão do próximo produto a ser fabricado.')

# Campos de entrada para o colaborador
st.header('Informações do Pedido e Situação Atual:')

# Entrada para Estoque_OK
estoque_ok_input = st.radio('O estoque de papel, tinta, tubete e caixa está OK para o produto desejado?', ('Sim', 'Nao'))

# Entrada para Dias_Desde_Pedido
dias_desde_pedido_input = st.slider('Quantos dias se passaram desde a entrada do pedido?', 1, 60, 5)

# Entrada para Urgencia_Pedido
urgencia_pedido_input = st.selectbox('Qual a urgência do pedido?', opcoes_urgencia)

# Entrada para Nome_Cliente
nome_cliente_input = st.selectbox('Qual o nome do Cliente?', opcoes_clientes)

# Entrada para Tempo_Medio_Producao_Min
tempo_medio_producao_input = st.slider('Tempo médio estimado de produção para este tipo de etiqueta (minutos)?', 20, 180, 60)


if st.button('Obter Sugestão de Produção'):
    # Preparar os dados de entrada do usuário para a previsão
    dados_entrada = pd.DataFrame({
        'Estoque_OK': [estoque_ok_input],
        'Dias_Desde_Pedido': [dias_desde_pedido_input],
        'Urgencia_Pedido': [urgencia_pedido_input],
        'Nome_Cliente': [nome_cliente_input],
        'Tempo_Medio_Producao_Min': [tempo_medio_producao_input]
    })

    # Aplicar One-Hot Encoding nos dados de entrada do usuário
    dados_entrada_processados = pd.get_dummies(dados_entrada, columns=[
        'Estoque_OK',
        'Urgencia_Pedido',
        'Nome_Cliente'
    ])

    # Garantir que as colunas correspondam às colunas de treinamento
    dados_entrada_final = dados_entrada_processados.reindex(columns=colunas_treinamento, fill_value=0)

    # Fazer a previsão
    previsao = model.predict(dados_entrada_final)

    st.success(f"A IA sugere produzir: **{previsao[0]}**")
    st.write("---")
    st.info("Lembre-se que esta é uma simulação. Para um uso real, você precisará de dados históricos da sua linha de produção.")