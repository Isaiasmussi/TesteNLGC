import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import requests

st.set_page_config(page_title="Executive Dashboard", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Roboto', sans-serif;
            color: #2c3e50;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        h1 {
            color: #0f293a;
            font-weight: 700;
            font-size: 2.5rem;
            border-bottom: 2px solid #b0bec5;
            padding-bottom: 10px;
        }
        h3 {
            color: #455a64;
            font-weight: 400;
            margin-top: 20px;
        }
        
        div.metric-container {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            text-align: center;
        }
        label.metric-label {
            font-size: 0.9rem !important;
            color: #78909c !important;
            text-transform: uppercase;
        }
        div.metric-value {
            font-size: 1.8rem !important;
            color: #0f293a !important;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data_from_api():
    api_url = "https://script.google.com/macros/s/AKfycbxHG51T-YJi8XpY1ZFmJ-YvNHO_OLxNA6TGp6BnUY_R539HsQW7bVpEth23TShRdqV1/exec"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        
        df_func = pd.DataFrame(data['funcionarios'])
        df_perf = pd.DataFrame(data['performance'])
        df_sal = pd.DataFrame(data['salarios'])

        for col in df_func.select_dtypes(include=['object']).columns:
            df_func[col] = df_func[col].str.strip()
            
        df_func['matricula'] = pd.to_numeric(df_func['matricula'], errors='coerce')
        df_perf['matricula'] = pd.to_numeric(df_perf['matricula'], errors='coerce')
        
        df = pd.merge(df_func, df_perf, on='matricula', how='inner')
        
        ref_date = pd.to_datetime('2025-12-02')
        df['Data de Admissão'] = pd.to_datetime(df['Data de Admissão'], errors='coerce')
        df['Data última promoção'] = pd.to_datetime(df['Data última promoção'], errors='coerce')
        
        df['Meses_Casa'] = ((ref_date - df['Data de Admissão']) / pd.Timedelta(days=30)).fillna(0)
        df['Meses_Sem_Promocao'] = ((ref_date - df['Data última promoção'].fillna(df['Data de Admissão'])) / pd.Timedelta(days=30)).fillna(0)
        
        return df, df_sal
        
    except Exception as e:
        st.error(f"Erro de conexão com a API: {e}")
        return None, None

df_full, df_sal = load_data_from_api()

if df_full is not None:
    df_sal['Valor'] = pd.to_numeric(df_sal['Valor'], errors='coerce')
    sal_map = df_sal.set_index('Nível de Cargo')['Valor'].to_dict()
    
    df_full['Salario_Atual'] = df_full['Nível de Cargo'].map(sal_map)
    promo_map = {'I': 'II', 'II': 'III', 'III': 'IV', 'IV': 'TETO'}
    df_full['Proximo_Nivel'] = df_full['Nível de Cargo'].map(promo_map)
    df_full['Salario_Novo'] = df_full['Proximo_Nivel'].map(sal_map)
    df_full['Custo_Aumento'] = df_full['Salario_Novo'] - df_full['Salario_Atual']
    df_process = df_full.dropna(subset=['Custo_Aumento']).copy()

    scaler = MinMaxScaler(feature_range=(0, 10))
    df_process['reincidencia_inv'] = 1 - df_process['reincidencia']
    cols_norm = ['tarefas', 'qualidade', 'reincidencia_inv']
    df_process[['n_t', 'n_q', 'n_r']] = scaler.fit_transform(df_process[cols_norm])
    df_process['Score_Tecnico'] = (0.4 * df_process['n_q']) + (0.3 * df_process['n_t']) + (0.3 * df_process['n_r'])

    st.sidebar.title("Painel de Controle")
    st.sidebar.markdown("**Parâmetros de Simulação**")
    budget_total = st.sidebar.number_input("Budget (R$)", 3000.0, step=100.0)
    min_fit = st.sidebar.slider("Corte Cultural", 0.0, 10.0, 8.0)
    min_tempo_casa = st.sidebar.slider("Min. Meses Casa", 0, 24, 6)
    
    df_process['Status'] = 'Não Elegível'
    mask_elegivel = (df_process['fit_cultural'] >= min_fit) & (df_process['Meses_Casa'] >= min_tempo_casa) & (df_process['Meses_Sem_Promocao'] >= 12)
    df_process.loc[mask_elegivel, 'Status'] = 'Elegível'
    
    mask_hold = (df_process['fit_cultural'] >= min_fit) & ((df_process['Meses_Casa'] < min_tempo_casa) | (df_process['Meses_Sem_Promocao'] < 12)) & (df_process['Score_Tecnico'] > 7)
    df_process.loc[mask_hold, 'Status'] = 'Retenção (Hold)'

    candidatos = df_process[df_process['Status'] == 'Elegível'].sort_values('Score_Tecnico', ascending=False).copy()
    candidatos['Custo_Acumulado'] = candidatos['Custo_Aumento'].cumsum()
    promovidos = candidatos[candidatos['Custo_Acumulado'] <= budget_total].copy()
    df_process.loc[df_process['matricula'].isin(promovidos['matricula']), 'Status'] = 'Promovido'

    st.markdown("<h1>Alocação Estratégica de Mérito</h1>", unsafe_allow_html=True)
    st.markdown("### Análise de Performance Técnica vs. Aderência Cultural")
    st.markdown("---")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    def style_metric(label, value, col):
        col.markdown(f"""
            <div class="metric-container">
                <label class="metric-label">{label}</label>
                <div class="metric-value">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    style_metric("Colaboradores Promovidos", len(promovidos), kpi1)
    style_metric("Investimento Total", f"R$ {promovidos['Custo_Aumento'].sum():.2f}", kpi2)
    style_metric("Utilização do Budget", f"{(promovidos['Custo_Aumento'].sum()/budget_total)*100:.1f}%", kpi3)
    style_metric("Média Score Técnico (Selecionados)", f"{promovidos['Score_Tecnico'].mean():.2f}", kpi4)

    st.markdown("<br>", unsafe_allow_html=True)

    col_chart, col_list = st.columns([1.8, 1])

    with col_chart:
        st.markdown("### Matriz de Decisão")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        colors = {'Não Elegível': '#cfd8dc', 'Retenção (Hold)': '#f0ad4e', 'Elegível': '#90a4ae', 'Promovido': '#2e7d32'}
        
        sns.scatterplot(data=df_process, x='Score_Tecnico', y='fit_cultural', hue='Status', palette=colors, s=100, alpha=0.8, ax=ax)
        
        for _, row in df_process[df_process['Status']=='Promovido'].iterrows():
            ax.text(row['Score_Tecnico']+0.1, row['fit_cultural'], f"{int(row['matricula'])}", fontsize=8, fontweight='bold', color='#1b5e20')

        ax.axhline(min_fit, color='#d32f2f', linestyle='--', linewidth=1)
        ax.text(2, min_fit+0.1, f"Corte Cultural ({min_fit})", color='#d32f2f', fontsize=8)

        ax.set_xlabel("Score Técnico (Produtividade + Qualidade)", fontsize=10)
        ax.set_ylabel("Fit Cultural (Avaliação Gestor)", fontsize=10)
        ax.legend(loc='lower left', frameon=True)
        ax.set_xlim(2, 10.5)
        ax.set_ylim(2, 10.5)
        
        st.pyplot(fig)

    with col_list:
        st.markdown("### Lista de Aprovação")
        view_df = promovidos[['matricula', 'Nível de Cargo', 'Proximo_Nivel', 'Score_Tecnico', 'Custo_Aumento']].copy()
        view_df.columns = ['ID', 'Nível Atual', 'Novo Nível', 'Score', 'Custo']
        
        st.dataframe(
            view_df.style.format({'Score': '{:.2f}', 'Custo': 'R$ {:.2f}'})
            .background_gradient(subset=['Score'], cmap='Greens'),
            use_container_width=True,
            height=400,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### Pontos de Atenção & Recomendações")
    st.info(f"""
    **Análise de Retenção:** Identificamos **{len(df_process[df_process['Status']=='Retenção (Hold)'])} colaboradores** com alta performance que não atingiram o tempo mínimo de casa. 
    Recomendamos feedback de reconhecimento para evitar frustração.
    """)
