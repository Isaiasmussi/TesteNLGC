import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

st.set_page_config(page_title="People Analytics Case", layout="wide")

st.title("📊 Dashboard de Promoção & Mérito")
st.markdown("""
**Contexto:** Ferramenta de apoio à decisão para alocação de mérito com base em Performance Técnica e Fit Cultural.
**Objetivo:** Maximizar o ROI do orçamento disponível promovendo os talentos certos.
""")

# --- GERAÇÃO DE DADOS MOCK (Para funcionamento sem arquivos locais) ---
@st.cache_data
def load_mock_data():
    np.random.seed(42)
    n_funcionarios = 50
    
    # DataFrame Funcionários
    data_func = {
        'matricula': range(1001, 1001 + n_funcionarios),
        'Data de Admissão': pd.date_range(start='2020-01-01', end='2024-01-01', periods=n_funcionarios),
        'Data última promoção': pd.to_datetime([
            np.random.choice([x, pd.NaT]) for x in pd.date_range(start='2022-01-01', end='2024-06-01', periods=n_funcionarios)
        ]),
        'Nível de Cargo': np.random.choice(['I', 'II', 'III', 'IV'], n_funcionarios, p=[0.3, 0.4, 0.2, 0.1])
    }
    df_func = pd.DataFrame(data_func)
    
    # DataFrame Performance
    data_perf = {
        'matricula': range(1001, 1001 + n_funcionarios),
        'tarefas': np.random.randint(50, 100, n_funcionarios),   # 0-100
        'qualidade': np.random.randint(60, 100, n_funcionarios), # 0-100
        'reincidencia': np.random.uniform(0, 0.2, n_funcionarios), # % de erro
        'fit_cultural': np.random.uniform(5.0, 10.0, n_funcionarios).round(1) # 0-10
    }
    df_perf = pd.DataFrame(data_perf)
    
    # Tabela Salarial
    df_sal = pd.DataFrame({
        'Nível de Cargo': ['I', 'II', 'III', 'IV', 'TETO'],
        'Valor': [3000, 4500, 6500, 9000, 12000]
    })
    
    # Merge inicial
    df = pd.merge(df_func, df_perf, on='matricula', how='inner')
    
    # Feature Engineering de Tempo
    ref_date = pd.to_datetime(datetime.now())
    df['Meses_Casa'] = ((ref_date - df['Data de Admissão']) / pd.Timedelta(days=30)).fillna(0).astype(int)
    
    # Lógica para data de promoção vazia (assume admissão)
    df['Data Ref Promo'] = df['Data última promoção'].fillna(df['Data de Admissão'])
    df['Meses_Sem_Promocao'] = ((ref_date - df['Data Ref Promo']) / pd.Timedelta(days=30)).fillna(0).astype(int)
    
    return df, df_sal

df_full, df_sal = load_mock_data()

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Parâmetros de Decisão")
budget_total = st.sidebar.number_input("Orçamento Disponível (R$)", value=15000.0, step=500.0)
min_fit = st.sidebar.slider("Fit Cultural Mínimo", 0.0, 10.0, 7.5, 0.1)
min_tempo_casa = st.sidebar.slider("Mínimo Meses de Casa", 0, 36, 12)
min_tempo_promo = st.sidebar.slider("Mínimo Meses s/ Promoção", 0, 24, 12)

# --- PROCESSAMENTO ---
if df_full is not None:
    # 1. Mapeamento Salarial
    sal_map = df_sal.set_index('Nível de Cargo')['Valor'].to_dict()
    df_full['Salario_Atual'] = df_full['Nível de Cargo'].map(sal_map)
    
    promo_map = {'I': 'II', 'II': 'III', 'III': 'IV', 'IV': 'TETO'}
    df_full['Proximo_Nivel'] = df_full['Nível de Cargo'].map(promo_map)
    df_full['Salario_Novo'] = df_full['Proximo_Nivel'].map(sal_map)
    df_full['Custo_Aumento'] = df_full['Salario_Novo'] - df_full['Salario_Atual']
    
    # Exclui quem já está no teto
    df_process = df_full.dropna(subset=['Custo_Aumento']).copy()

    # 2. Score Técnico (Normalização Simplificada)
    # Normalizando entre 0 e 10 manualmente para evitar dependência do sklearn no exemplo simples
    def normalize(series):
        return (series - series.min()) / (series.max() - series.min()) * 10
    
    df_process['n_t'] = normalize(df_process['tarefas'])
    df_process['n_q'] = normalize(df_process['qualidade'])
    df_process['n_r'] = normalize(1 - df_process['reincidencia']) # Inverso pois reincidencia é ruim
    
    df_process['Score_Tecnico'] = (0.4 * df_process['n_q']) + (0.3 * df_process['n_t']) + (0.3 * df_process['n_r'])
    df_process['Score_Tecnico'] = df_process['Score_Tecnico'].fillna(0)

    # 3. Status e Regras
    df_process['Status'] = 'Não Elegível'

    # Elegíveis
    mask_elegivel = (
        (df_process['fit_cultural'] >= min_fit) & 
        (df_process['Meses_Casa'] >= min_tempo_casa) & 
        (df_process['Meses_Sem_Promocao'] >= min_tempo_promo)
    )
    df_process.loc[mask_elegivel, 'Status'] = 'Elegível'

    # Hold (High Potentials barrados por tempo)
    mask_barrado = (
        (df_process['fit_cultural'] >= min_fit) & 
        ((df_process['Meses_Casa'] < min_tempo_casa) | (df_process['Meses_Sem_Promocao'] < min_tempo_promo)) & 
        (df_process['Score_Tecnico'] > 7.0)
    )
    df_process.loc[mask_barrado, 'Status'] = 'Hold (Tempo)'

    # 4. Distribuição do Budget (Algoritmo Guloso)
    candidatos = df_process[df_process['Status'] == 'Elegível'].sort_values('Score_Tecnico', ascending=False).copy()
    candidatos['Custo_Acumulado'] = candidatos['Custo_Aumento'].cumsum()
    
    promovidos = candidatos[candidatos['Custo_Acumulado'] <= budget_total].copy()
    ids_promovidos = promovidos['matricula'].tolist()
    
    df_process.loc[df_process['matricula'].isin(ids_promovidos), 'Status'] = 'Promovido'

    # --- KPI DISPLAY ---
    col1, col2, col3, col4 = st.columns(4)
    custo_promo = promovidos['Custo_Aumento'].sum()
    
    col1.metric("Pessoas Promovidas", len(promovidos))
    col2.metric("Custo Total", f"R$ {custo_promo:,.2f}")
    col3.metric("Sobras do Budget", f"R$ {(budget_total - custo_promo):,.2f}", 
                delta_color="normal" if (budget_total - custo_promo) > 0 else "inverse")
    
    score_medio = promovidos['Score_Tecnico'].mean() if not promovidos.empty else 0
    col4.metric("Score Técnico Médio", f"{score_medio:.2f}")

    st.divider()

    # --- VISUALIZAÇÃO (ALTAIR) ---
    col_graf, col_tab = st.columns([2, 1])

    with col_graf:
        st.subheader("Matriz de Decisão (Interativa)")
        
        # Cores customizadas
        domain = ['Não Elegível', 'Hold (Tempo)', 'Elegível', 'Promovido']
        range_ = ['#e0e0e0', '#f39c12', '#95a5a6', '#27ae60']
        
        chart = alt.Chart(df_process).mark_circle(size=100).encode(
            x=alt.X('Score_Tecnico', title='Score Técnico (0-10)', scale=alt.Scale(domain=[0, 10])),
            y=alt.Y('fit_cultural', title='Fit Cultural (0-10)', scale=alt.Scale(domain=[0, 10])),
            color=alt.Color('Status', scale=alt.Scale(domain=domain, range=range_), legend=alt.Legend(title="Status Final")),
            tooltip=['matricula', 'Nível de Cargo', 'Meses_Casa', 'Score_Tecnico', 'fit_cultural', 'Custo_Aumento']
        ).interactive()

        # Linha de corte do Fit
        rule = alt.Chart(pd.DataFrame({'y': [min_fit]})).mark_rule(color='red', strokeDash=[5, 5]).encode(y='y')
        
        st.altair_chart(chart + rule, use_container_width=True)

    with col_tab:
        st.subheader("Lista Final")
        if not promovidos.empty:
            display_cols = ['matricula', 'Nível de Cargo', 'Score_Tecnico', 'Custo_Aumento']
            st.dataframe(
                promovidos[display_cols].style.format({
                    'Score_Tecnico': '{:.2f}', 
                    'Custo_Aumento': 'R$ {:.2f}'
                }), 
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Ninguém foi promovido com os critérios atuais.")

    # --- WATCHLIST ---
    st.divider()
    st.subheader("🚩 Watchlist & Riscos")
    
    col_risk1, col_risk2 = st.columns(2)
    
    with col_risk1:
        st.markdown("**High Potentials em 'Hold' (Risco de Saída)**")
        df_hold = df_process[df_process['Status'] == 'Hold (Tempo)']
        if not df_hold.empty:
            st.dataframe(df_hold[['matricula', 'Score_Tecnico', 'Meses_Casa', 'Meses_Sem_Promocao']])
        else:
            st.success("Nenhum talento retido por tempo.")

    with col_risk2:
        st.markdown("**Distribuição por Nível**")
        if not df_process.empty:
            chart_bar = alt.Chart(df_process).mark_bar().encode(
                x='Nível de Cargo',
                y='count()',
                color='Status'
            )
            st.altair_chart(chart_bar, use_container_width=True)
