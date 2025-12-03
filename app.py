import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="People Analytics Case", layout="wide")

# Título e Contexto
st.title("📊 Dashboard de Promoção & Mérito")
st.markdown("""
**Contexto:** Ferramenta de apoio à decisão para alocação de mérito com base em Performance Técnica e Fit Cultural.
**Objetivo:** Maximizar o ROI do orçamento disponível promovendo os talentos certos.
""")

# --- CARREGAMENTO DE DADOS ---
@st.cache_data
def load_data():
    # Carregando os arquivos (Assumindo que estarão na mesma pasta no GitHub)
    try:
        df_func = pd.read_csv('basededadosnelogica.xlsx - Funcionário.csv')
        df_perf = pd.read_csv('basededadosnelogica.xlsx - Performance.csv')
        df_sal = pd.read_csv('basededadosnelogica.xlsx - Tabela Salarial.csv')
        
        # Limpeza básica
        for col in df_func.select_dtypes(include=['object']).columns:
            df_func[col] = df_func[col].str.strip()
            
        df_func['matricula'] = pd.to_numeric(df_func['matricula'], errors='coerce')
        df_perf['matricula'] = pd.to_numeric(df_perf['matricula'], errors='coerce')
        
        # Merge
        df = pd.merge(df_func, df_perf, on='matricula', how='inner')
        
        # Datas (Ref: 02/12/2025)
        ref_date = pd.to_datetime('2025-12-02')
        df['Data de Admissão'] = pd.to_datetime(df['Data de Admissão'], errors='coerce')
        df['Data última promoção'] = pd.to_datetime(df['Data última promoção'], errors='coerce')
        
        # Feature Engineering de Tempo
        df['Meses_Casa'] = ((ref_date - df['Data de Admissão']) / pd.Timedelta(days=30)).fillna(0)
        df['Meses_Sem_Promocao'] = ((ref_date - df['Data última promoção'].fillna(df['Data de Admissão'])) / pd.Timedelta(days=30)).fillna(0)
        
        return df, df_sal
    except Exception as e:
        st.error(f"Erro ao carregar arquivos: {e}")
        return None, None

df_full, df_sal = load_data()

if df_full is not None:
    # --- BARRA LATERAL (CONTROLES) ---
    st.sidebar.header("⚙️ Parâmetros de Decisão")
    budget_total = st.sidebar.number_input("Orçamento Disponível (R$)", value=3000.0, step=100.0)
    min_fit = st.sidebar.slider("Fit Cultural Mínimo", 0.0, 10.0, 8.0, 0.5)
    min_tempo_casa = st.sidebar.slider("Mínimo Meses de Casa", 0, 24, 6)
    min_tempo_promo = st.sidebar.slider("Mínimo Meses s/ Promoção", 0, 24, 12)

    # --- PROCESSAMENTO (REGRAS DE NEGÓCIO) ---
    # 1. Salários e Custos
    df_sal['Valor'] = pd.to_numeric(df_sal['Valor'], errors='coerce')
    sal_map = df_sal.set_index('Nível de Cargo')['Valor'].to_dict()
    
    df_full['Salario_Atual'] = df_full['Nível de Cargo'].map(sal_map)
    promo_map = {'I': 'II', 'II': 'III', 'III': 'IV', 'IV': 'TETO'}
    df_full['Proximo_Nivel'] = df_full['Nível de Cargo'].map(promo_map)
    df_full['Salario_Novo'] = df_full['Proximo_Nivel'].map(sal_map)
    df_full['Custo_Aumento'] = df_full['Salario_Novo'] - df_full['Salario_Atual']
    
    # Filtrar apenas quem tem próximo nível (Exclui Teto)
    df_process = df_full.dropna(subset=['Custo_Aumento']).copy()

    # 2. Score Técnico
    scaler = MinMaxScaler(feature_range=(0, 10))
    df_process['reincidencia_inv'] = 1 - df_process['reincidencia']
    cols_norm = ['tarefas', 'qualidade', 'reincidencia_inv']
    norm_data = scaler.fit_transform(df_process[cols_norm])
    df_process[['n_t', 'n_q', 'n_r']] = norm_data
    
    # Pesos
    df_process['Score_Tecnico'] = (0.4 * df_process['n_q']) + (0.3 * df_process['n_t']) + (0.3 * df_process['n_r'])

    # 3. Definição de Status
    df_process['Status'] = 'Não Elegível'
    
    # Regra de Elegibilidade
    mask_elegivel = (
        (df_process['fit_cultural'] >= min_fit) & 
        (df_process['Meses_Casa'] >= min_tempo_casa) & 
        (df_process['Meses_Sem_Promocao'] >= min_tempo_promo)
    )
    df_process.loc[mask_elegivel, 'Status'] = 'Elegível'
    
    # Regra de "Barrados pelo Tempo" (High Potential)
    mask_barrado = (
        (df_process['fit_cultural'] >= min_fit) & 
        ((df_process['Meses_Casa'] < min_tempo_casa) | (df_process['Meses_Sem_Promocao'] < min_tempo_promo)) & 
        (df_process['Score_Tecnico'] > 7)
    )
    df_process.loc[mask_barrado, 'Status'] = 'Hold (Tempo)'

    # 4. Seleção Orçamentária
    candidatos = df_process[df_process['Status'] == 'Elegível'].sort_values('Score_Tecnico', ascending=False).copy()
    candidatos['Custo_Acumulado'] = candidatos['Custo_Aumento'].cumsum()
    promovidos = candidatos[candidatos['Custo_Acumulado'] <= budget_total].copy()
    
    ids_promovidos = promovidos['matricula'].tolist()
    df_process.loc[df_process['matricula'].isin(ids_promovidos), 'Status'] = 'Promovido'

    # --- DASHBOARD VISUAL ---
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Colaboradores Promovidos", len(promovidos))
    col2.metric("Custo Total", f"R$ {promovidos['Custo_Aumento'].sum():.2f}")
    col3.metric("Sobras do Budget", f"R$ {(budget_total - promovidos['Custo_Aumento'].sum()):.2f}")
    col4.metric("Score Técnico Médio (Promovidos)", f"{promovidos['Score_Tecnico'].mean():.2f}")

    st.markdown("---")

    # Gráfico Principal
    col_graf, col_tab = st.columns([2, 1])

    with col_graf:
        st.subheader("Matriz de Decisão (9-Box Adaptado)")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot Layers
        sns.scatterplot(data=df_process[df_process['Status']=='Não Elegível'], x='Score_Tecnico', y='fit_cultural', 
                        color='lightgray', s=50, alpha=0.5, label='Outros', ax=ax)
        sns.scatterplot(data=df_process[df_process['Status']=='Hold (Tempo)'], x='Score_Tecnico', y='fit_cultural', 
                        color='orange', s=100, marker='s', label='Hold (Tempo)', ax=ax)
        sns.scatterplot(data=df_process[df_process['Status']=='Elegível'], x='Score_Tecnico', y='fit_cultural', 
                        color='gray', s=80, label='Elegível (Sem Budget)', ax=ax)
        sns.scatterplot(data=df_process[df_process['Status']=='Promovido'], x='Score_Tecnico', y='fit_cultural', 
                        color='#27ae60', s=200, edgecolor='black', label='PROMOVIDO', ax=ax)

        # Labels
        for _, row in df_process[df_process['Status']=='Promovido'].iterrows():
            ax.text(row['Score_Tecnico']+0.1, row['fit_cultural'], f"ID {int(row['matricula'])}", fontweight='bold')

        # Linhas de Corte
        ax.axhline(min_fit, color='red', linestyle='--', alpha=0.5, label=f'Corte Fit ({min_fit})')
        
        ax.set_title("Performance Técnica vs Alinhamento Cultural", fontsize=12)
        ax.set_xlabel("Score Técnico (0-10)")
        ax.set_ylabel("Fit Cultural")
        ax.legend(loc='lower left')
        st.pyplot(fig)

    with col_tab:
        st.subheader("Lista Final de Promoção")
        st.dataframe(
            promovidos[['matricula', 'Nível de Cargo', 'Proximo_Nivel', 'Meses_Casa', 'Score_Tecnico', 'Custo_Aumento']]
            .style.format({'Score_Tecnico': '{:.2f}', 'Custo_Aumento': 'R$ {:.2f}', 'Meses_Casa': '{:.1f}'}),
            use_container_width=True,
            hide_index=True
        )

    # --- ANÁLISE DETALHADA ---
    st.markdown("---")
    st.subheader("🚩 Pontos de Atenção (Watchlist)")
    col_hold1, col_hold2 = st.columns(2)
    
    with col_hold1:
        st.markdown("**Top Talents Barrados por Tempo (Risco de Retenção):**")
        st.dataframe(df_process[df_process['Status']=='Hold (Tempo)'][['matricula', 'Score_Tecnico', 'Meses_Casa', 'Meses_Sem_Promocao']])
        
    with col_hold2:
        st.markdown("**Análise por Nível de Senioridade (Média):**")
        nivel_stats = df_process.groupby('Nível de Cargo', observed=False)[['Score_Tecnico', 'fit_cultural']].mean()
        st.dataframe(nivel_stats.style.background_gradient(cmap='Blues'))
