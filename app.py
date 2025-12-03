import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime

# Configuração da Página
st.set_page_config(page_title="People Analytics - Assistente de Suporte", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS CSS (Design Corporativo Limpo) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; color: #2c3e50; }
        
        /* Ajuste de espaçamento */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        /* Cards de Métricas */
        div.metric-container {
            background-color: #ffffff; border-left: 5px solid #2c3e50; padding: 15px;
            border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        label.metric-label { font-size: 0.85rem !important; color: #7f8c8d !important; text-transform: uppercase; letter-spacing: 1px; }
        div.metric-value { font-size: 1.8rem !important; color: #2c3e50 !important; font-weight: 700; }
        
        /* Títulos e Âncoras */
        h1, h2, h3 { color: #2c3e50; font-weight: 700; }
        
        /* Links do Índice Lateral */
        a.toc-link { color: #34495e; text-decoration: none; font-size: 0.95rem; display: block; padding: 5px 0; }
        a.toc-link:hover { color: #2980b9; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- 1. CARREGAMENTO DE DADOS ---
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

# --- ÍNDICE LATERAL (Manual) ---
st.sidebar.markdown("### Índice")
st.sidebar.markdown("""
- [Premissas & Metodologia](#premissas-metodologia)
- [Dashboard de Decisão](#dashboard-de-decisao)
- [Lista Final & Insights](#lista-final-insights)
""", unsafe_allow_html=True)


if df_func is not None and not df_func.empty and not df_perf.empty:

    # --- PROCESSAMENTO DOS DADOS ---
    df_func['matricula'] = df_func['matricula'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df_perf['matricula'] = df_perf['matricula'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    df = pd.merge(df_func, df_perf, on='matricula', how='inner')
    
    col_admissao = 'Data de Admissão'
    if col_admissao in df.columns:
        df[col_admissao] = pd.to_datetime(df[col_admissao], errors='coerce')
        agora = pd.Timestamp.now()
        df['dias_casa'] = (agora - df[col_admissao]).dt.days
        df['Meses_Casa'] = (df['dias_casa'] / 30.0).fillna(0).astype(int)
    else:
        df['Meses_Casa'] = 0

    if not df_sal.empty and 'Nível de Cargo' in df.columns:
        df_sal['Nível de Cargo'] = df_sal['Nível de Cargo'].astype(str).str.strip()
        df['Nível de Cargo'] = df['Nível de Cargo'].astype(str).str.strip()
        
        if df_sal['Valor'].dtype == 'O': 
            df_sal['Valor'] = df_sal['Valor'].astype(str).str.replace('R$', '', regex=False)
            df_sal['Valor'] = df_sal['Valor'].str.replace('.', '', regex=False) 
            df_sal['Valor'] = df_sal['Valor'].str.replace(',', '.', regex=False) 
            
        df_sal['Valor'] = pd.to_numeric(df_sal['Valor'], errors='coerce')
        df_sal_map = df_sal.groupby('Nível de Cargo')['Valor'].mean().to_dict()
        
        df['Salario_Atual'] = df['Nível de Cargo'].map(df_sal_map)
        mapa_promocao = {'I': 'II', 'II': 'III', 'III': 'IV', 'IV': 'TETO'}
        df['Proximo_Nivel'] = df['Nível de Cargo'].map(mapa_promocao)
        df['Salario_Novo'] = df['Proximo_Nivel'].map(df_sal_map)
        df['Custo_Aumento'] = df['Salario_Novo'] - df['Salario_Atual']
        df_elegiveis = df.dropna(subset=['Custo_Aumento']).copy()
    else:
        st.error("Erro nos dados salariais.")
        st.stop()

    if df_elegiveis.empty: st.stop()

    cols_calc = ['tarefas', 'qualidade', 'reincidencia', 'avaliacao_gestor', 'fit_cultural']
    for col in cols_calc:
        if col in df_elegiveis.columns:
            df_elegiveis[col] = df_elegiveis[col].astype(str).str.replace(',', '.')
            df_elegiveis[col] = pd.to_numeric(df_elegiveis[col], errors='coerce').fillna(0)

    # Cálculo Scores (Fórmula Nelogica)
    max_tarefas = df_elegiveis['tarefas'].max()
    if max_tarefas == 0: max_tarefas = 1
    df_elegiveis['nota_produtividade'] = (df_elegiveis['tarefas'] / max_tarefas) * 10

    if df_elegiveis['reincidencia'].max() > 1.0:
        df_elegiveis['reincidencia'] = df_elegiveis['reincidencia'] / 100.0
    df_elegiveis['nota_eficiencia'] = (1 - df_elegiveis['reincidencia']) * 10
    df_elegiveis['nota_eficiencia'] = df_elegiveis['nota_eficiencia'].clip(0, 10)

    if df_elegiveis['qualidade'].max() > 10: df_elegiveis['qualidade'] /= 10.0
    if df_elegiveis['avaliacao_gestor'].max() > 10: df_elegiveis['avaliacao_gestor'] /= 10.0

    df_elegiveis['Score_Tecnico'] = (df_elegiveis['qualidade'] * 0.30) + \
                                    (df_elegiveis['nota_produtividade'] * 0.30) + \
                                    (df_elegiveis['avaliacao_gestor'] * 0.20) + \
                                    (df_elegiveis['nota_eficiencia'] * 0.20)

    # --- SEÇÃO 1: TÍTULO E PREMISSAS ---
    st.title("People Analytics - Assistente de Suporte")
    
    st.header("Premissas & Metodologia", anchor="premissas-metodologia")
    st.info("Este painel utiliza um algoritmo de decisão multicritério para garantir promoções meritocráticas, transparentes e alinhadas à cultura.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Critérios de Elegibilidade")
        st.markdown("""
        * **Fit Cultural >= 8.0:** Obrigatório. Alinhamento aos valores da empresa.
        * **Tempo de Casa >= 12 Meses:** Maturação necessária para o próximo nível.
        """)
    
    with col2:
        st.subheader("2. Algoritmo de Performance")
        st.markdown("Score Final (0 a 10) composto por:")
        st.markdown("""
        * **30% Produtividade (Volume):** Normalizado pelo máximo do time.
        * **30% Qualidade (CSAT):** Foco na experiência do cliente.
        * **20% Eficiência (Zero Erros):** Penaliza a reincidência.
        * **20% Avaliação do Gestor:** Soft skills e liderança.
        """)
    
    st.markdown("---")

    # --- SEÇÃO 2: DASHBOARD (COM CONTROLES EMBUTIDOS) ---
    st.header("Dashboard de Decisão", anchor="dashboard-de-decisao")
    
    # Controles dentro da página (Expander para não poluir)
    with st.expander("Configurações do Cenário (Budget e Filtros)", expanded=True):
        c1, c2 = st.columns(2)
        budget_total = c1.number_input("Budget Disponível (R$)", value=3000.0, step=100.0)
        fit_corte = c2.slider("Régua Fit Cultural", 8.0, 10.0, 8.0)

    # Lógica de Seleção
    mask_promocao = (
        (df_elegiveis['fit_cultural'] >= fit_corte) & 
        (df_elegiveis['Meses_Casa'] >= 12)
    )
    candidatos = df_elegiveis[mask_promocao].copy().sort_values(by='Score_Tecnico', ascending=False)
    candidatos['Custo_Acumulado'] = candidatos['Custo_Aumento'].cumsum()
    promovidos = candidatos[candidatos['Custo_Acumulado'] <= budget_total].copy()
    
    df_elegiveis['Status'] = 'Não Elegível'
    df_elegiveis.loc[mask_promocao, 'Status'] = 'Elegível (Sem Budget)' 
    df_elegiveis.loc[df_elegiveis['Meses_Casa'] < 12, 'Status'] = 'Em Maturação (<12m)'
    df_elegiveis.loc[df_elegiveis['matricula'].isin(promovidos['matricula']), 'Status'] = 'PROMOVIDO'

    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    uso_budget = (promovidos['Custo_Aumento'].sum() / budget_total * 100) if budget_total > 0 else 0
    
    kpi1.markdown(f'<div class="metric-container"><label class="metric-label">Colaboradores Promovidos</label><div class="metric-value">{len(promovidos)}</div></div>', unsafe_allow_html=True)
    kpi2.markdown(f'<div class="metric-container"><label class="metric-label">Investimento Total</label><div class="metric-value">R$ {promovidos["Custo_Aumento"].sum():.2f}</div></div>', unsafe_allow_html=True)
    kpi3.markdown(f'<div class="metric-container"><label class="metric-label">Uso do Budget</label><div class="metric-value">{uso_budget:.1f}%</div></div>', unsafe_allow_html=True)
    
    st.write("") # Espaçamento

    # Gráfico
    col_chart, col_table = st.columns([1.8, 1])
    
    with col_chart:
        st.markdown("##### Matriz de Performance x Cultura")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        sns.scatterplot(data=df_elegiveis[~df_elegiveis['Status'].isin(['PROMOVIDO', 'Em Maturação (<12m)'])], 
                        x='Score_Tecnico', y='fit_cultural', color='#95a5a6', alpha=0.4, s=60, label='Outros', ax=ax)
        sns.scatterplot(data=df_elegiveis[df_elegiveis['Status'] == 'Em Maturação (<12m)'], 
                        x='Score_Tecnico', y='fit_cultural', color='#f39c12', alpha=0.6, s=80, marker='X', label='< 12 Meses', ax=ax)
        
        if not promovidos.empty:
            sns.scatterplot(data=promovidos, x='Score_Tecnico', y='fit_cultural', 
                            color='#2ecc71', s=150, edgecolor='#27ae60', label='Promovidos', ax=ax)
            for line in range(0, promovidos.shape[0]):
                ax.text(promovidos.Score_Tecnico.iloc[line]+0.05, promovidos.fit_cultural.iloc[line], 
                        f"ID {promovidos.matricula.iloc[line]}", horizontalalignment='left', size='small', color='black', weight='bold')
            ax.axvline(x=promovidos['Score_Tecnico'].min(), color='#3498db', linestyle='--', alpha=0.5, label='Corte Dinâmico')

        ax.axhline(y=fit_corte, color='#e74c3c', linestyle='--', alpha=0.5, label=f'Régua Fit ({fit_corte})')
        ax.legend(loc='lower left', frameon=True)
        ax.set_xlabel("Score Técnico")
        ax.set_ylabel("Fit Cultural")
        st.pyplot(fig, use_container_width=True)

    with col_table:
        st.markdown("##### Lista Final")
        if not promovidos.empty:
            st.dataframe(
                promovidos[['matricula', 'Proximo_Nivel', 'Score_Tecnico', 'tarefas']].rename(columns={'tarefas': 'Vol.', 'Proximo_Nivel': 'Cargo Novo'})
                .style.format({'Score_Tecnico': '{:.2f}', 'Vol.': '{:.0f}'})
                .background_gradient(subset=['Score_Tecnico'], cmap='Greens'),
                use_container_width=True, height=400, hide_index=True
            )
        else:
            st.warning("Nenhum colaborador atingiu os critérios.")

    st.markdown("---")

    # --- SEÇÃO 3: INSIGHTS ---
    st.header("Lista Final & Insights", anchor="lista-final-insights")
    
    if not promovidos.empty:
        top_performer = promovidos.iloc[0]
        avg_score_prom = promovidos['Score_Tecnico'].mean()
        avg_score_geral = df_elegiveis['Score_Tecnico'].mean()
        
        st.markdown(f"""
        **Análise das Promoções**
        
        Com base nos critérios estabelecidos, selecionamos **{len(promovidos)} colaboradores** que combinam alta entrega técnica e forte alinhamento cultural.
        
        **Destaque do Ciclo: Colaborador {top_performer['matricula']}**
        O colaborador de matrícula **{top_performer['matricula']}** obteve a maior pontuação global (**{top_performer['Score_Tecnico']:.2f}**).
        * Volume: Entregou **{top_performer['tarefas']:.0f}** tarefas.
        * Eficiência: Apresentou uma taxa de reincidência de apenas **{top_performer['reincidencia']:.2f}%**.
        
        **Elevação da Barra**
        * A média de Score Técnico dos promovidos foi de **{avg_score_prom:.2f}**.
        * Isso representa um desempenho **{((avg_score_prom/avg_score_geral)-1)*100:.1f}% superior** à média geral da equipe ({avg_score_geral:.2f}).
        
        **Pontos de Atenção (Maturação)**
        Identificamos **{len(df_elegiveis[df_elegiveis['Status'] == 'Em Maturação (<12m)'])} colaboradores** com alto potencial (Fit e Técnica), mas que foram retidos pela regra de 12 meses de casa. Recomenda-se feedback de retenção para estes talentos (pontos laranja no gráfico).
        """)
    else:
        st.info("Não há dados suficientes de promoções para gerar insights neste cenário.")

else:
    st.info("Carregando dados da API...")
