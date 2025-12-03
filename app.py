import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import requests

st.set_page_config(page_title="Executive Dashboard - Nelogica Case", layout="wide", initial_sidebar_state="collapsed")

# --- ESTILIZAÇÃO CSS ---
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

# --- CARREGAMENTO DE DADOS ---
@st.cache_data(ttl=600)
def load_data_from_api():
    api_url = "https://script.google.com/macros/s/AKfycbxHG51T-YJi8XpY1ZFmJ-YvNHO_OLxNA6TGp6BnUY_R539HsQW7bVpEth23TShRdqV1/exec"
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        # Função auxiliar para achar chaves ignorando maiúsculas/minúsculas
        def get_df(data_dict, keys):
            for k in keys:
                if k in data_dict: return pd.DataFrame(data_dict[k])
            return pd.DataFrame()

        df_func = get_df(data, ['funcionarios', 'Funcionário', 'Funcionario'])
        df_perf = get_df(data, ['performance', 'Performance'])
        df_sal = get_df(data, ['salarios', 'tabela_salarial', 'Tabela Salarial'])

        if df_func.empty or df_perf.empty: return None, None

        # Converter matrículas para garantir o merge
        for d in [df_func, df_perf]:
            if 'matricula' in d.columns:
                d['matricula'] = pd.to_numeric(d['matricula'], errors='coerce')
        
        df = pd.merge(df_func, df_perf, on='matricula', how='inner')
        return df, df_sal

    except Exception as e:
        st.error(f"Erro na API: {e}")
        return None, None

df, df_sal = load_data_from_api()

if df is not None and not df.empty:
    
    # --- 2. ENGENHARIA DE SALÁRIOS ---
    if not df_sal.empty:
        df_sal['Valor'] = pd.to_numeric(df_sal['Valor'], errors='coerce')
        df_sal_map = df_sal.set_index('Nível de Cargo')['Valor'].to_dict()
        
        df['Salario_Atual'] = df['Nível de Cargo'].map(df_sal_map)
        mapa_promocao = {'I': 'II', 'II': 'III', 'III': 'IV', 'IV': 'TETO'}
        df['Proximo_Nivel'] = df['Nível de Cargo'].map(mapa_promocao)
        df['Salario_Novo'] = df['Proximo_Nivel'].map(df_sal_map)
        df['Custo_Aumento'] = df['Salario_Novo'] - df['Salario_Atual']
    else:
        df['Custo_Aumento'] = 0

    # Filtra elegíveis (quem tem aumento calculado)
    df_elegiveis = df.dropna(subset=['Custo_Aumento']).copy()

    # --- 3. SANITIZAÇÃO DE DADOS (CRUCIAL PARA CORRIGIR O ERRO) ---
    # Colunas que serão usadas no cálculo matemático
    cols_math = ['tarefas', 'qualidade', 'reincidencia', 'fit_cultural']
    
    # Verifica se as colunas existem, se não, cria com 0
    for col in cols_math:
        if col not in df_elegiveis.columns:
            df_elegiveis[col] = 0
        else:
            # 1. Converte para string
            # 2. Troca vírgula por ponto (caso venha "9,5")
            # 3. Força conversão para número (coerce transforma erros em NaN)
            # 4. Preenche NaN com 0
            df_elegiveis[col] = df_elegiveis[col].astype(str).str.replace(',', '.')
            df_elegiveis[col] = pd.to_numeric(df_elegiveis[col], errors='coerce').fillna(0)

    # --- 4. CÁLCULO DO SCORE TÉCNICO ---
    # Inverter reincidência (1 - valor). Como já garantimos que é número acima, não dará erro.
    df_elegiveis['reincidencia_score'] = 1 - df_elegiveis['reincidencia']

    # Normalização
    scaler = MinMaxScaler(feature_range=(0, 10))
    cols_norm = ['tarefas', 'qualidade', 'reincidencia_score']
    
    # Garante mais uma vez que não há NaNs antes de passar pro Scikit-Learn
    df_elegiveis[cols_norm] = df_elegiveis[cols_norm].fillna(0)
    
    try:
        dados_norm = scaler.fit_transform(df_elegiveis[cols_norm])
        df_norm = pd.DataFrame(dados_norm, columns=[c+'_n' for c in cols_norm], index=df_elegiveis.index)
        df_elegiveis = pd.concat([df_elegiveis, df_norm], axis=1)

        # Cálculo Score Ponderado
        df_elegiveis['Score_Tecnico'] = (df_elegiveis['qualidade_n'] * 0.4) + \
                                        (df_elegiveis['tarefas_n'] * 0.3) + \
                                        (df_elegiveis['reincidencia_score_n'] * 0.3)
    except Exception as e:
        st.error(f"Erro crítico no cálculo do Score: {e}")
        st.stop()

    # --- SIDEBAR E FILTROS ---
    st.sidebar.header("Painel de Decisão")
    budget_input = st.sidebar.number_input("Budget Disponível (R$)", value=3000.0, step=100.0)
    fit_corte = st.sidebar.slider("Corte Fit Cultural", 0, 10, 8)

    # --- 5. LÓGICA DE SELEÇÃO (FILTER & SORT) ---
    candidatos = df_elegiveis[df_elegiveis['fit_cultural'] >= fit_corte].copy()
    candidatos = candidatos.sort_values(by='Score_Tecnico', ascending=False)
    
    candidatos['Custo_Acumulado'] = candidatos['Custo_Aumento'].cumsum()
    promovidos = candidatos[candidatos['Custo_Acumulado'] <= budget_input].copy()

    # Status para visualização
    df_elegiveis['Status'] = 'Não Elegível'
    df_elegiveis.loc[df_elegiveis['fit_cultural'] >= fit_corte, 'Status'] = 'Elegível (Budget Insuficiente)'
    df_elegiveis.loc[df_elegiveis['matricula'].isin(promovidos['matricula']), 'Status'] = 'PROMOVIDO'

    # --- DASHBOARD ---
    st.markdown("### Alocação de Mérito (Regra: Fit Cultural + Score Técnico)")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.markdown(f'<div class="metric-container"><label class="metric-label">Promovidos</label><div class="metric-value">{len(promovidos)}</div></div>', unsafe_allow_html=True)
    kpi2.markdown(f'<div class="metric-container"><label class="metric-label">Investimento</label><div class="metric-value">R$ {promovidos["Custo_Aumento"].sum():.2f}</div></div>', unsafe_allow_html=True)
    
    perc_budget = (promovidos["Custo_Aumento"].sum() / budget_input * 100) if budget_input > 0 else 0
    kpi3.markdown(f'<div class="metric-container"><label class="metric-label">Uso do Budget</label><div class="metric-value">{perc_budget:.1f}%</div></div>', unsafe_allow_html=True)
    
    avg_score = promovidos['Score_Tecnico'].mean() if not promovidos.empty else 0
    kpi4.markdown(f'<div class="metric-container"><label class="metric-label">Média Score (Sel.)</label><div class="metric-value">{avg_score:.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    col_grafico, col_tabela = st.columns([2, 1])

    with col_grafico:
        st.markdown("#### Matriz de Decisão")
        if not df_elegiveis.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.set_style("whitegrid")
            
            # Plot dos não promovidos
            sns.scatterplot(data=df_elegiveis[df_elegiveis['Status'] != 'PROMOVIDO'], 
                            x='Score_Tecnico', y='fit_cultural', 
                            color='grey', alpha=0.3, s=80, ax=ax)
            
            # Plot dos promovidos
            if not promovidos.empty:
                sns.scatterplot(data=promovidos, x='Score_Tecnico', y='fit_cultural', 
                                color='#2ecc71', s=200, edgecolor='black', label='Promovidos', ax=ax)
                
                for _, row in promovidos.iterrows():
                    ax.text(row['Score_Tecnico']+0.1, row['fit_cultural'], 
                            f"ID {int(row['matricula'])}", fontsize=8, weight='bold')

            ax.axhline(y=fit_corte, color='r', linestyle='--', alpha=0.5, label=f'Corte Cultural ({fit_corte})')
            ax.set_xlabel('Score Técnico Final')
            ax.set_ylabel('Fit Cultural')
            ax.legend()
            st.pyplot(fig)

    with col_tabela:
        st.markdown("#### Lista Final")
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
            st.warning("Nenhum colaborador elegível.")

else:
    st.info("Aguardando carregamento dos dados...")
