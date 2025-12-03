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

# --- ESTILOS CSS (Design Dashboard & Índice) ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; color: #e0e0e0; background-color: #0e1117; }
        
        /* Remover padding excessivo do topo */
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }

        /* Estilo do Índice Lateral (Colab Style) */
        [data-testid="stSidebar"] {
            background-color: #161b22;
            border-right: 1px solid #30363d;
        }
        .toc-header {
            font-size: 1.5rem; font-weight: 700; color: #f0f6fc; margin-bottom: 1rem;
            display: flex; align-items: center; justify-content: space-between;
        }
        .toc-link {
            display: block; padding: 8px 0; color: #8b949e; text-decoration: none;
            font-size: 0.95rem; transition: color 0.2s;
        }
        .toc-link:hover { color: #58a6ff; }
        .toc-active { color: #f0f6fc; font-weight: 500; border-left: 2px solid #58a6ff; padding-left: 10px; }
        
        /* Cards de Conteúdo (Quadro Transparente/Glassmorphism) */
        .dashboard-card {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            backdrop-filter: blur(5px);
        }
        
        /* Títulos dentro dos cards */
        .card-title {
            color: #58a6ff; font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;
            text-transform: uppercase; letter-spacing: 1px;
        }
        
        /* Texto corrido */
        .card-text { font-size: 0.95rem; line-height: 1.6; color: #c9d1d9; }
        
        /* Métricas */
        div.metric-container {
            background-color: #0d1117; border: 1px solid #30363d; padding: 15px;
            border-radius: 6px; text-align: center;
        }
        label.metric-label { font-size: 0.8rem !important; color: #8b949e !important; text-transform: uppercase; }
        div.metric-value { font-size: 1.6rem !important; color: #f0f6fc !important; font-weight: 700; }

        /* Separadores */
        hr { border-color: #30363d; margin: 30px 0; }
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

# --- ÍNDICE LATERAL (ESTILO COLAB) ---
st.sidebar.markdown("""
<div class="toc-header">
    Índice 
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8b949e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3h18v18H3zM3 9h18M9 21V9"/></svg>
</div>
<div style="margin-left: 5px;">
    <a href="#premissas" class="toc-link">1. Premissas & Metodologia</a>
    <a href="#dashboard" class="toc-link">2. Dashboard de Performance</a>
    <a href="#insights" class="toc-link">3. Insights Gerenciais</a>
</div>
<div style="margin-top: 30px; font-size: 0.8rem; color: #484f58;">
    © 2025 People Analytics
</div>
""", unsafe_allow_html=True)

if df_func is not None and not df_func.empty and not df_perf.empty:

    # --- PROCESSAMENTO (VALORES FIXOS: FIT 8.0, BUDGET 3000) ---
    FIT_CORTE = 8.0
    BUDGET_TOTAL = 3000.0

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

    mask_promocao = (
        (df_elegiveis['fit_cultural'] >= FIT_CORTE) & 
        (df_elegiveis['Meses_Casa'] >= 12)
    )
    candidatos = df_elegiveis[mask_promocao].copy().sort_values(by='Score_Tecnico', ascending=False)
    candidatos['Custo_Acumulado'] = candidatos['Custo_Aumento'].cumsum()
    promovidos = candidatos[candidatos['Custo_Acumulado'] <= BUDGET_TOTAL].copy()
    
    df_elegiveis['Status'] = 'Não Elegível'
    df_elegiveis.loc[mask_promocao, 'Status'] = 'Elegível (Sem Budget)' 
    df_elegiveis.loc[df_elegiveis['Meses_Casa'] < 12, 'Status'] = 'Em Maturação (<12m)'
    df_elegiveis.loc[df_elegiveis['matricula'].isin(promovidos['matricula']), 'Status'] = 'PROMOVIDO'

    # --- CORPO DO DASHBOARD ---
    
    st.title("People Analytics | Matriz de Decisão")
    
    # SEÇÃO 1: PREMISSAS (EM CARD)
    st.markdown('<div id="premissas"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">1. Premissas & Metodologia</div>
        <div class="card-text">
            Este modelo utiliza um algoritmo multicritério para garantir meritocracia.
            <br><br>
            <strong>Critérios de Corte (Gatekeepers):</strong><br>
            • <strong>Fit Cultural ≥ 8.0:</strong> Obrigatório para garantir alinhamento aos valores.<br>
            • <strong>Tempo de Casa ≥ 12 Meses:</strong> Maturação necessária para o cargo.
            <br><br>
            <strong>Composição do Score Técnico (0-10):</strong><br>
            • <strong>30% Produtividade:</strong> Volume de tarefas (normalizado pelo máximo do time).<br>
            • <strong>30% Qualidade:</strong> Satisfação do cliente (CSAT).<br>
            • <strong>20% Eficiência:</strong> Baixa taxa de reincidência.<br>
            • <strong>20% Avaliação Gestor:</strong> Soft skills e comportamento.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SEÇÃO 2: DASHBOARD (GRÁFICO + TABELA)
    st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)
    
    # KPIs
    kpi1, kpi2, kpi3 = st.columns(3)
    uso_budget = (promovidos['Custo_Aumento'].sum() / BUDGET_TOTAL * 100) if BUDGET_TOTAL > 0 else 0
    
    kpi1.markdown(f'<div class="metric-container"><label class="metric-label">Promovidos</label><div class="metric-value">{len(promovidos)}</div></div>', unsafe_allow_html=True)
    kpi2.markdown(f'<div class="metric-container"><label class="metric-label">Investimento</label><div class="metric-value">R$ {promovidos["Custo_Aumento"].sum():.2f}</div></div>', unsafe_allow_html=True)
    kpi3.markdown(f'<div class="metric-container"><label class="metric-label">Budget Utilizado</label><div class="metric-value">{uso_budget:.1f}%</div></div>', unsafe_allow_html=True)
    
    st.write("")

    col_chart, col_table = st.columns([1.6, 1])
    
    with col_chart:
        st.markdown("##### Performance x Cultura")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Ajustando tema escuro para o gráfico combinar com o dashboard
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        
        # Cores customizadas para tema escuro
        sns.scatterplot(data=df_elegiveis[~df_elegiveis['Status'].isin(['PROMOVIDO', 'Em Maturação (<12m)'])], 
                        x='Score_Tecnico', y='fit_cultural', color='#30363d', alpha=0.6, s=60, label='Outros', ax=ax)
        sns.scatterplot(data=df_elegiveis[df_elegiveis['Status'] == 'Em Maturação (<12m)'], 
                        x='Score_Tecnico', y='fit_cultural', color='#d29922', alpha=0.7, s=80, marker='X', label='< 12 Meses', ax=ax)
        
        if not promovidos.empty:
            sns.scatterplot(data=promovidos, x='Score_Tecnico', y='fit_cultural', 
                            color='#238636', s=150, edgecolor='#f0f6fc', linewidth=1.5, label='Promovidos', ax=ax)
            for line in range(0, promovidos.shape[0]):
                ax.text(promovidos.Score_Tecnico.iloc[line]+0.08, promovidos.fit_cultural.iloc[line], 
                        f"ID {promovidos.matricula.iloc[line]}", horizontalalignment='left', size='small', color='#f0f6fc', weight='bold')
            ax.axvline(x=promovidos['Score_Tecnico'].min(), color='#1f6feb', linestyle='--', alpha=0.6, label='Corte Dinâmico')

        ax.axhline(y=FIT_CORTE, color='#da3633', linestyle='--', alpha=0.6, label=f'Régua Fit ({FIT_CORTE})')
        
        # Ajuste de eixos para tema escuro
        ax.tick_params(colors='#8b949e')
        ax.xaxis.label.set_color('#8b949e')
        ax.yaxis.label.set_color('#8b949e')
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('#30363d') 
        ax.spines['right'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')

        legend = ax.legend(loc='lower left', frameon=True, facecolor='#161b22', edgecolor='#30363d')
        plt.setp(legend.get_texts(), color='#8b949e')
        
        ax.set_xlabel("Score Técnico")
        ax.set_ylabel("Fit Cultural")
        st.pyplot(fig, use_container_width=True)

    with col_table:
        st.markdown("##### Lista Final")
        if not promovidos.empty:
            st.dataframe(
                promovidos[['matricula', 'Proximo_Nivel', 'Score_Tecnico', 'tarefas']].rename(columns={'tarefas': 'Vol.', 'Proximo_Nivel': 'Novo Cargo'}),
                use_container_width=True, height=400, hide_index=True
            )
        else:
            st.warning("Nenhum colaborador elegível.")

    # SEÇÃO 3: INSIGHTS (EM CARD)
    st.markdown('<div id="insights"></div>', unsafe_allow_html=True)
    
    if not promovidos.empty:
        top_performer = promovidos.iloc[0]
        avg_score_prom = promovidos['Score_Tecnico'].mean()
        avg_score_geral = df_elegiveis['Score_Tecnico'].mean()
        
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">3. Insights Gerenciais</div>
            <div class="card-text">
                Com base nos dados processados:<br><br>
                <strong>🏆 Destaque do Ciclo: Colaborador {top_performer['matricula']}</strong><br>
                Atingiu o maior Score Global (<strong>{top_performer['Score_Tecnico']:.2f}</strong>), combinando um volume alto de <strong>{top_performer['tarefas']:.0f} tarefas</strong> com uma taxa de erro mínima (<strong>{top_performer['reincidencia']:.2f}%</strong>).<br><br>
                <strong>📈 Elevação da Barra Técnica</strong><br>
                O grupo de promovidos performa <strong>{((avg_score_prom/avg_score_geral)-1)*100:.1f}% acima</strong> da média geral da equipe.<br><br>
                <strong>⚠️ Pipeline de Talentos</strong><br>
                Existem <strong>{len(df_elegiveis[df_elegiveis['Status'] == 'Em Maturação (<12m)'])} colaboradores</strong> com performance de promoção, mas retidos pelo tempo de casa (pontos amarelos no gráfico). Acompanhar para o próximo ciclo.
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Carregando dados da API...")
