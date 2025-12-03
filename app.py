import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import requests

st.set_page_config(page_title="Executive Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- ESTILOS CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; color: #2c3e50; }
        div.metric-container {
            background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px;
            border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center;
        }
        label.metric-label { font-size: 0.9rem !important; color: #78909c !important; text-transform: uppercase; }
        div.metric-value { font-size: 1.8rem !important; color: #0f293a !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 1. CARREGAMENTO E TRATAMENTO INICIAL ---
@st.cache_data(ttl=600)
def load_data():
    api_url = "https://script.google.com/macros/s/AKfycbxHG51T-YJi8XpY1ZFmJ-YvNHO_OLxNA6TGp6BnUY_R539HsQW7bVpEth23TShRdqV1/exec"
    try:
        r = requests.get(api_url)
        r.raise_for_status()
        data = r.json()
        
        # Função helper para pegar tabelas
        def get_df(d, keys):
            for k in keys: 
                if k in d: return pd.DataFrame(d[k])
            return pd.DataFrame()

        df_func = get_df(data, ['funcionarios', 'Funcionário', 'Funcionario'])
        df_perf = get_df(data, ['performance', 'Performance'])
        df_sal = get_df(data, ['salarios', 'tabela_salarial', 'Tabela Salarial'])

        return df_func, df_perf, df_sal
    except Exception as e:
        st.error(f"Erro na API: {e}")
        return None, None, None

df_func, df_perf, df_sal = load_data()

if df_func is not None and not df_func.empty and not df_perf.empty:

    # --- GARANTIA DE MERGE (Matrículas) ---
    # Converte tudo para texto, remove .0 decimal e remove espaços
    df_func['matricula'] = df_func['matricula'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df_perf['matricula'] = df_perf['matricula'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    # Merge das bases
    df = pd.merge(df_func, df_perf, on='matricula', how='inner')
    
    # --- 2. ENGENHARIA DE SALÁRIOS E CUSTOS ---
    if not df_sal.empty:
        # Limpeza crucial para o mapeamento funcionar (Remove espaços de "II ", " III")
        df_sal['Nível de Cargo'] = df_sal['Nível de Cargo'].astype(str).str.strip()
        df['Nível de Cargo'] = df['Nível de Cargo'].astype(str).str.strip()
        
        # Converte valores monetários
        df_sal['Valor'] = pd.to_numeric(df_sal['Valor'], errors='coerce')
        
        # Mapear valores salariais atuais
        df_sal_map = df_sal.set_index('Nível de Cargo')['Valor'].to_dict()
        df['Salario_Atual'] = df['Nível de Cargo'].map(df_sal_map)

        # Definir próximo nível e custo
        mapa_promocao = {'I': 'II', 'II': 'III', 'III': 'IV', 'IV': 'TETO'}
        df['Proximo_Nivel'] = df['Nível de Cargo'].map(mapa_promocao)
        
        # Calcula salário novo
        df['Salario_Novo'] = df['Proximo_Nivel'].map(df_sal_map)
        df['Custo_Aumento'] = df['Salario_Novo'] - df['Salario_Atual']
        
        # Remover quem já está no teto (IV) ou falhou no mapeamento
        # Se aqui der zero linhas, é porque os nomes dos cargos (I, II...) não bateram
        df_elegiveis = df.dropna(subset=['Custo_Aumento']).copy()
    else:
        st.error("Tabela salarial vazia ou inválida.")
        st.stop()

    # Checagem de segurança antes do cálculo matemático
    if df_elegiveis.empty:
        st.error("ERRO: Nenhuma promoção calculável.")
        st.code(f"Linhas no Merge: {len(df)}\nLinhas após cálculo salarial: 0\n\nPossível erro: Os nomes dos cargos na tabela Funcionário não batem com a Tabela Salarial.")
        st.write("Amostra Cargo Funcionário:", df['Nível de Cargo'].unique())
        st.write("Amostra Cargo Tabela Salarial:", df_sal['Nível de Cargo'].unique())
        st.stop()

    # --- 3. ENGENHARIA DO SCORE TÉCNICO (0 a 10) ---
    # Tratamento de vírgulas e conversão para número
    cols_calc = ['tarefas', 'qualidade', 'reincidencia', 'fit_cultural']
    for col in cols_calc:
        if col in df_elegiveis.columns:
            df_elegiveis[col] = df_elegiveis[col].astype(str).str.replace(',', '.')
            df_elegiveis[col] = pd.to_numeric(df_elegiveis[col], errors='coerce').fillna(0)

    scaler = MinMaxScaler(feature_range=(0, 10))

    # Inverter reincidência (quanto menor, melhor)
    df_elegiveis['reincidencia_score'] = 1 - df_elegiveis['reincidencia']

    # Normalizar métricas
    cols_norm = ['tarefas', 'qualidade', 'reincidencia_score']
    
    # Aqui dava o erro de 0 samples se a tabela estivesse vazia. Agora já bloqueamos antes.
    dados_norm = scaler.fit_transform(df_elegiveis[cols_norm])
    df_norm = pd.DataFrame(dados_norm, columns=[c+'_n' for c in cols_norm], index=df_elegiveis.index)
    
    df_elegiveis = pd.concat([df_elegiveis, df_norm], axis=1)

    # Cálculo Score Técnico (Média Ponderada: 40% Qualidade, 30% Volume, 30% Baixa Reincidência)
    df_elegiveis['Score_Tecnico'] = (df_elegiveis['qualidade_n'] * 0.4) + \
                                    (df_elegiveis['tarefas_n'] * 0.3) + \
                                    (df_elegiveis['reincidencia_score_n'] * 0.3)

    # --- 4. DASHBOARD E SIDEBAR ---
    st.sidebar.title("Painel de Controle")
    budget_total = st.sidebar.number_input("Budget (R$)", value=3000.0, step=100.0)
    fit_corte = st.sidebar.slider("Corte Fit Cultural", 0.0, 10.0, 8.0)

    # Regra de Negócio: Seleção (9-Box: Alto Potencial)
    candidatos = df_elegiveis[(df_elegiveis['fit_cultural'] >= fit_corte)].copy()

    # Ordenar pelo Score Técnico (Mérito)
    candidatos = candidatos.sort_values(by='Score_Tecnico', ascending=False)

    # Seleção Orçamentária
    candidatos['Custo_Acumulado'] = candidatos['Custo_Aumento'].cumsum()
    promovidos = candidatos[candidatos['Custo_Acumulado'] <= budget_total].copy()

    # Marcar status na tabela principal para o gráfico
    df_elegiveis['Status'] = 'Não Elegível'
    df_elegiveis.loc[df_elegiveis['fit_cultural'] >= fit_corte, 'Status'] = 'Elegível (Budget Insuficiente)'
    df_elegiveis.loc[df_elegiveis['matricula'].isin(promovidos['matricula']), 'Status'] = 'PROMOVIDO'

    # --- VISUALIZAÇÃO ---
    st.markdown("### Matriz de Decisão: Performance x Cultura")
    
    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.markdown(f'<div class="metric-container"><label class="metric-label">Promovidos</label><div class="metric-value">{len(promovidos)}</div></div>', unsafe_allow_html=True)
    kpi2.markdown(f'<div class="metric-container"><label class="metric-label">Investimento</label><div class="metric-value">R$ {promovidos["Custo_Aumento"].sum():.2f}</div></div>', unsafe_allow_html=True)
    uso_budget = (promovidos['Custo_Aumento'].sum() / budget_total * 100) if budget_total > 0 else 0
    kpi3.markdown(f'<div class="metric-container"><label class="metric-label">Uso do Budget</label><div class="metric-value">{uso_budget:.1f}%</div></div>', unsafe_allow_html=True)
    
    st.markdown("---")

    col_chart, col_table = st.columns([1.8, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")

        # Todos os pontos (Cinza)
        sns.scatterplot(data=df_elegiveis[df_elegiveis['Status'] != 'PROMOVIDO'], 
                        x='Score_Tecnico', y='fit_cultural', 
                        color='grey', alpha=0.3, s=60, label='Elegíveis/Resto')

        # Promovidos (Verde)
        if not promovidos.empty:
            sns.scatterplot(data=promovidos, x='Score_Tecnico', y='fit_cultural', 
                            color='#2ecc71', s=150, edgecolor='black', label='Promover (Top Pick)')

            # Anotações
            for line in range(0, promovidos.shape[0]):
                ax.text(promovidos.Score_Tecnico.iloc[line]+0.1,
                        promovidos.fit_cultural.iloc[line],
                        f"ID {int(promovidos.matricula.iloc[line])}",
                        horizontalalignment='left', size='small', color='black', weight='semibold')
            
            # Linha de corte técnico dinâmico
            ax.axvline(x=promovidos['Score_Tecnico'].min(), color='b', linestyle='--', alpha=0.5, label='Corte Técnico Dinâmico')

        ax.axhline(y=fit_corte, color='r', linestyle='--', alpha=0.5, label=f'Corte Cultural ({fit_corte})')
        
        ax.set_title('Matriz de Decisão', fontsize=12)
        ax.set_xlabel('Score Técnico (Volume + Qualidade + Baixa Reincidência)', fontsize=10)
        ax.set_ylabel('Fit Cultural (Avaliação Gestor)', fontsize=10)
        ax.legend(loc='lower left')
        st.pyplot(fig)

    with col_table:
        st.markdown("#### Lista de Promoção")
        if not promovidos.empty:
            st.dataframe(
                promovidos[['matricula', 'Nível de Cargo', 'Proximo_Nivel', 'Score_Tecnico', 'Custo_Aumento']]
                .style.format({'Score_Tecnico': '{:.2f}', 'Custo_Aumento': 'R$ {:.2f}', 'matricula': '{:.0f}'})
                .background_gradient(subset=['Score_Tecnico'], cmap='Greens'),
                use_container_width=True,
                height=400,
                hide_index=True
            )
        else:
            st.warning("Ninguém elegível dentro do budget.")

else:
    st.info("Aguardando carregamento da API...")
