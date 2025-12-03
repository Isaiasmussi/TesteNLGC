import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime

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

# --- 1. CARREGAMENTO ---
@st.cache_data(ttl=600)
def load_data():
    api_url = "https://script.google.com/macros/s/AKfycbxHG51T-YJi8XpY1ZFmJ-YvNHO_OLxNA6TGp6BnUY_R539HsQW7bVpEth23TShRdqV1/exec"
    try:
        r = requests.get(api_url)
        r.raise_for_status()
        data = r.json()
        
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

    # Tratamento de chaves
    df_func['matricula'] = df_func['matricula'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df_perf['matricula'] = df_perf['matricula'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    df = pd.merge(df_func, df_perf, on='matricula', how='inner')
    
    # --- CORREÇÃO DO ERRO DE DATA (TELA VERMELHA) ---
    col_admissao = 'Data de Admissão'
    if col_admissao in df.columns:
        df[col_admissao] = pd.to_datetime(df[col_admissao], errors='coerce')
        agora = pd.Timestamp.now()
        # Cálculo seguro: Dias totais / média de dias no mês
        df['dias_casa'] = (agora - df[col_admissao]).dt.days
        df['Meses_Casa'] = (df['dias_casa'] / 30.44).fillna(0).astype(int)
    else:
        st.error(f"Coluna '{col_admissao}' não encontrada.")
        df['Meses_Casa'] = 0

    # --- 2. ENGENHARIA DE SALÁRIOS ---
    if not df_sal.empty and 'Nível de Cargo' in df.columns:
        # Padronização para Texto
        df_sal['Nível de Cargo'] = df_sal['Nível de Cargo'].astype(str).str.strip()
        df['Nível de Cargo'] = df['Nível de Cargo'].astype(str).str.strip()
        
        df_sal['Valor'] = pd.to_numeric(df_sal['Valor'], errors='coerce')
        df_sal_map = df_sal.groupby('Nível de Cargo')['Valor'].mean().to_dict()
        
        df['Salario_Atual'] = df['Nível de Cargo'].map(df_sal_map)
        
        # --- MAPA DE PROMOÇÃO (AQUI PODE ESTAR O ERRO DA BASE VAZIA) ---
        # Verifique se seus níveis são I, II, III ou 1, 2, 3 ou Jr, Pl, Sr
        mapa_promocao = {'I': 'II', 'II': 'III', 'III': 'IV', 'IV': 'TETO'}
        
        df['Proximo_Nivel'] = df['Nível de Cargo'].map(mapa_promocao)
        df['Salario_Novo'] = df['Proximo_Nivel'].map(df_sal_map)
        df['Custo_Aumento'] = df['Salario_Novo'] - df['Salario_Atual']
        
        df_elegiveis = df.dropna(subset=['Custo_Aumento']).copy()
    else:
        st.error("Tabelas salariais ou coluna 'Nível de Cargo' ausentes.")
        st.stop()

    # --- DIAGNÓSTICO DE BASE VAZIA ---
    if df_elegiveis.empty:
        st.warning("⚠️ Base vazia após cálculo salarial. O cruzamento falhou.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Níveis encontrados nos Funcionários:**")
            st.write(df['Nível de Cargo'].unique())
        
        with col2:
            st.markdown("**Níveis encontrados na Tabela Salarial:**")
            st.write(list(df_sal_map.keys()))
            
        st.info("💡 **Dica:** Os nomes acima precisam ser IDÊNTICOS. Se um for 'Analista I' e o outro for só 'I', o sistema não cruza. Ajuste o 'mapa_promocao' no código se necessário.")
        st.stop()

    # --- 3. SCORE TÉCNICO (40/30/30) ---
    cols_calc = ['tarefas', 'qualidade', 'reincidencia', 'fit_cultural']
    for col in cols_calc:
        if col in df_elegiveis.columns:
            df_elegiveis[col] = df_elegiveis[col].astype(str).str.replace(',', '.')
            df_elegiveis[col] = pd.to_numeric(df_elegiveis[col], errors='coerce').fillna(0)

    scaler = MinMaxScaler(feature_range=(0, 10))
    df_elegiveis['reincidencia_score'] = df_elegiveis['reincidencia'] * -1 

    cols_norm = ['tarefas', 'qualidade', 'reincidencia_score']
    dados_norm = scaler.fit_transform(df_elegiveis[cols_norm])
    df_norm = pd.DataFrame(dados_norm, columns=[c+'_n' for c in cols_norm], index=df_elegiveis.index)
    df_elegiveis = pd.concat([df_elegiveis, df_norm], axis=1)

    df_elegiveis['Score_Tecnico'] = (df_elegiveis['qualidade_n'] * 0.40) + \
                                    (df_elegiveis['tarefas_n'] * 0.30) + \
                                    (df_elegiveis['reincidencia_score_n'] * 0.30)

    # --- 4. DASHBOARD ---
    st.sidebar.title("Painel de Controle")
    budget_total = st.sidebar.number_input("Budget (R$)", value=3000.0, step=100.0)
    fit_corte = st.sidebar.slider("Régua Fit Cultural", 8.0, 10.0, 8.0) 

    mask_promocao = (
        (df_elegiveis['fit_cultural'] >= fit_corte) & 
        (df_elegiveis['Meses_Casa'] >= 12)
    )
    
    candidatos = df_elegiveis[mask_promocao].copy()
    candidatos = candidatos.sort_values(by='Score_Tecnico', ascending=False)

    candidatos['Custo_Acumulado'] = candidatos['Custo_Aumento'].cumsum()
    promovidos = candidatos[candidatos['Custo_Acumulado'] <= budget_total].copy()

    df_elegiveis['Status'] = 'Não Elegível'
    df_elegiveis.loc[mask_promocao, 'Status'] = 'Elegível (Budget Insuficiente)' 
    df_elegiveis.loc[df_elegiveis['Meses_Casa'] < 12, 'Status'] = 'Em Maturação (<12m)'
    df_elegiveis.loc[df_elegiveis['matricula'].isin(promovidos['matricula']), 'Status'] = 'PROMOVIDO'

    st.markdown("### Matriz de Decisão: Performance x Cultura")
    
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.markdown(f'<div class="metric-container"><label class="metric-label">Promovidos</label><div class="metric-value">{len(promovidos)}</div></div>', unsafe_allow_html=True)
    kpi2.markdown(f'<div class="metric-container"><label class="metric-label">Investimento</label><div class="metric-value">R$ {promovidos["Custo_Aumento"].sum():.2f}</div></div>', unsafe_allow_html=True)
    uso_budget = (promovidos['Custo_Aumento'].sum() / budget_total * 100) if budget_total > 0 else 0
    kpi3.markdown(f'<div class="metric-container"><label class="metric-label">Uso do Budget</label><div class="metric-value">{uso_budget:.1f}%</div></div>', unsafe_allow_html=True)
    st.markdown("---")

    col_chart, col_table = st.columns([1.8, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.set_style("whitegrid")

        # Plots
        sns.scatterplot(data=df_elegiveis[~df_elegiveis['Status'].isin(['PROMOVIDO', 'Em Maturação (<12m)'])], 
                        x='Score_Tecnico', y='fit_cultural', color='grey', alpha=0.3, s=60, label='Outros', ax=ax)
        
        sns.scatterplot(data=df_elegiveis[df_elegiveis['Status'] == 'Em Maturação (<12m)'], 
                        x='Score_Tecnico', y='fit_cultural', color='orange', alpha=0.5, s=80, marker='X', label='< 12 Meses', ax=ax)

        if not promovidos.empty:
            sns.scatterplot(data=promovidos, x='Score_Tecnico', y='fit_cultural', 
                            color='#2ecc71', s=150, edgecolor='black', label='Promovidos', ax=ax)
            for line in range(0, promovidos.shape[0]):
                ax.text(promovidos.Score_Tecnico.iloc[line]+0.05, promovidos.fit_cultural.iloc[line], 
                        f"ID {promovidos.matricula.iloc[line]}", horizontalalignment='left', size='small', color='black', weight='semibold')
            ax.axvline(x=promovidos['Score_Tecnico'].min(), color='b', linestyle='--', alpha=0.5)

        ax.axhline(y=fit_corte, color='r', linestyle='--', alpha=0.5, label=f'Régua ({fit_corte})')
        ax.legend(loc='lower left', frameon=True)
        ax.set_xlabel("Score Técnico (0-10)")
        ax.set_ylabel("Fit Cultural (0-10)")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col_table:
        st.markdown("#### Lista de Promoção")
        if not promovidos.empty:
            st.dataframe(promovidos[['matricula', 'Meses_Casa', 'Proximo_Nivel', 'Score_Tecnico', 'Custo_Aumento']].style.format({'Score_Tecnico': '{:.2f}', 'Custo_Aumento': 'R$ {:.2f}'}).background_gradient(subset=['Score_Tecnico'], cmap='Greens'), use_container_width=True, height=450, hide_index=True)
        else:
            st.warning("Ninguém elegível.")
else:
    st.info("Carregando...")
