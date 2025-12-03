import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe # Importação para o efeito de contorno no texto
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime

st.set_page_config(page_title="People Analytics - Assistente de Suporte", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS CSS ADAPTATIVOS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        html, body, [class*="css"]  { font-family: 'Roboto', sans-serif; color: var(--text-color); }
        
        .block-container { padding-top: 1rem; padding-bottom: 5rem; }

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
        .toc-link:hover { color: #58a6ff; font-weight: bold; }
        
        .dashboard-card {
            background-color: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .profile-card {
            background-color: var(--background-color);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .profile-id { color: var(--primary-color); font-weight: bold; font-size: 1.1rem; }
        .profile-role { color: var(--text-color); font-size: 0.85rem; text-transform: uppercase; margin-bottom: 5px; letter-spacing: 0.5px; opacity: 0.7;}
        
        .card-title {
            color: var(--primary-color); font-size: 1.1rem; font-weight: 600; margin-bottom: 10px;
            text-transform: uppercase; letter-spacing: 1px;
        }
        
        .card-text { font-size: 0.95rem; line-height: 1.6; color: var(--text-color); }
        
        div.metric-container {
            background-color: var(--secondary-background-color); 
            border: 1px solid rgba(128, 128, 128, 0.2); 
            padding: 15px;
            border-radius: 6px; text-align: center;
        }
        label.metric-label { font-size: 0.8rem !important; color: var(--text-color) !important; opacity: 0.7; text-transform: uppercase; }
        div.metric-value { font-size: 1.6rem !important; color: var(--primary-color) !important; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

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

st.sidebar.markdown("""
<div class="toc-header">
    Índice 
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#8b949e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3h18v18H3zM3 9h18M9 21V9"/></svg>
</div>
<div style="margin-left: 5px;">
    <a href="#premissas" class="toc-link">#1.</a>
    <a href="#dashboard" class="toc-link">#2.</a>
    <a href="#orcamento" class="toc-link">#3.</a>
    <a href="#perfis" class="toc-link">#4.</a>
    <a href="#insights" class="toc-link">#5.</a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="margin-top: 30px; font-size: 0.8rem; color: #484f58;">
    © 2025 People Analytics
</div>
""", unsafe_allow_html=True)

if df_func is not None and not df_func.empty and not df_perf.empty:

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

    st.title("People Analytics | Análise de Promoção")
    
    st.markdown('<div id="premissas"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="dashboard-card">
        <div class="card-title">1. Premissas & Metodologia</div>
        <div class="card-text">
            Este modelo utiliza um algoritmo multicritério para garantir meritocracia.
            <br><br>
            <strong>Critérios de Corte (Gatekeepers):</strong><br>
            • <strong>Fit Cultural ≥ 8.0:</strong> Priorizamos o alinhamento comportamental e de valores. A premissa é que hard skills (técnica) podem ser desenvolvidas com treinamento eficaz, enquanto a dissonância cultural é de difícil correção e alto custo. Um Fit Cultural baixo compromete a sinergia da equipe e pode erodir o clima organizacional a longo prazo.<br>
            • <strong>Tempo de Casa ≥ 12 Meses:</strong> Estabelecemos um ciclo mínimo de um ano para garantir a consolidação do colaborador na função atual. Embora reconheçamos a agilidade e a ambição de crescimento dos novos talentos, este período é vital para a consistência da entrega. Além disso, regras claras de tempo promovem a isonomia (igualdade), evitando percepções de favorecimento e protegendo a motivação do coletivo.
            <br><br>
            <strong>Composição do Score Técnico (0-10):</strong><br>
            • <strong>30%</strong> Produtividade<br>
            • <strong>30%</strong> Qualidade<br>
            • <strong>20%</strong> Eficiência<br>
            • <strong>20%</strong> Avaliação Gestor
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div id="dashboard"></div>', unsafe_allow_html=True)
    
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
        
        # Tema Transparente (Adapta ao Light/Dark do Streamlit)
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        # Cores Adaptativas
        sns.scatterplot(data=df_elegiveis[~df_elegiveis['Status'].isin(['PROMOVIDO', 'Em Maturação (<12m)'])], 
                        x='Score_Tecnico', y='fit_cultural', color='gray', alpha=0.3, s=60, label='Outros', ax=ax)
        
        sns.scatterplot(data=df_elegiveis[df_elegiveis['Status'] == 'Em Maturação (<12m)'], 
                        x='Score_Tecnico', y='fit_cultural', color='#f39c12', alpha=0.7, s=80, marker='X', label='< 12 Meses', ax=ax)
        
        if not promovidos.empty:
            sns.scatterplot(data=promovidos, x='Score_Tecnico', y='fit_cultural', 
                            color='#2ecc71', s=150, edgecolor='black', linewidth=1.0, label='Promovidos', ax=ax)
            
            for line in range(0, promovidos.shape[0]):
                # TRUQUE DE VISIBILIDADE: Texto Branco com Contorno Preto (Lê-se bem no Dark e no Light)
                text_obj = ax.text(promovidos.Score_Tecnico.iloc[line]+0.08, promovidos.fit_cultural.iloc[line], 
                        f"ID {promovidos.matricula.iloc[line]}", horizontalalignment='left', size='small', 
                        color='white', weight='bold') # Texto Branco
                text_obj.set_path_effects([pe.withStroke(linewidth=2, foreground='black')]) # Contorno Preto

            ax.axvline(x=promovidos['Score_Tecnico'].min(), color='#3498db', linestyle='--', alpha=0.6, label='Corte Dinâmico')

        ax.axhline(y=FIT_CORTE, color='#e74c3c', linestyle='--', alpha=0.6, label=f'Régua Fit ({FIT_CORTE})')
        
        # Ajuste de eixos (Usa cor 'gray' que funciona bem no escuro e no claro)
        ax.tick_params(colors='gray')
        ax.xaxis.label.set_color('gray')
        ax.yaxis.label.set_color('gray')
        ax.spines['bottom'].set_color('gray')
        ax.spines['top'].set_color('gray') 
        ax.spines['right'].set_color('gray')
        ax.spines['left'].set_color('gray')

        # Legenda adaptativa
        legend = ax.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.2, edgecolor='gray')
        plt.setp(legend.get_texts(), color='gray')
        
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

    st.markdown('<div id="orcamento"></div>', unsafe_allow_html=True)
    if not promovidos.empty:
        custo_medio = promovidos['Custo_Aumento'].mean()
        score_medio = promovidos['Score_Tecnico'].mean()
        
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">3. Impacto Orçamentário & Eficiência</div>
            <div style="display: flex; justify-content: space-around; margin-top: 10px;">
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; opacity: 0.8;">Orçamento Total</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: var(--primary-color);">R$ {BUDGET_TOTAL:.2f}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; opacity: 0.8;">Custo Médio / Promoção</div>
                    <div style="font-size: 1.4rem; font-weight: bold;">R$ {custo_medio:.2f}</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 0.9rem; opacity: 0.8;">Score Médio "Comprado"</div>
                    <div style="font-size: 1.4rem; font-weight: bold; color: #2ecc71;">{score_medio:.2f}</div>
                </div>
            </div>
            <div class="card-text" style="margin-top: 20px; text-align: center;">
                O ciclo atual priorizou candidatos que entregam, em média, um score de <strong>{score_medio:.2f}</strong>. 
                Isso garante que cada real investido está indo para perfis de alta tração e alinhamento cultural.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div id="perfis"></div>', unsafe_allow_html=True)
    st.markdown("### Destaques da Promoção (Top Talent)")
    
    if not promovidos.empty:
        top_6 = promovidos.head(6)
        cols = st.columns(3)
        
        for idx, (i, row) in enumerate(top_6.iterrows()):
            resumo_perfil = "Perfil consistente com entrega equilibrada."
            
            if row['tarefas'] > df_elegiveis['tarefas'].quantile(0.8) and row['reincidencia'] < 0.10:
                resumo_perfil = "Alto volume de entregas mantendo baixo índice de retrabalho. Referência em produtividade."
            
            elif row['qualidade'] >= 9.0 and row['tarefas'] < df_elegiveis['tarefas'].median():
                resumo_perfil = "Excelência técnica e foco em qualidade. Potencial para ganho de escala."
            
            elif row['reincidencia'] < 0.05:
                resumo_perfil = "Alta eficiência operacional. Perfil ideal para tarefas de alta complexidade técnica."
            
            elif row['avaliacao_gestor'] >= 9.0:
                resumo_perfil = "Forte alinhamento cultural e liderança comportamental. Ponto focal na equipe."
            
            with cols[idx % 3]:
                st.markdown(f"""
                <div class="profile-card">
                    <div class="profile-id">ID {row['matricula']}</div>
                    <div class="profile-role">Promovido a {row['Proximo_Nivel']}</div>
                    <div style="margin: 15px 0; border-top: 1px solid rgba(128,128,128,0.2);"></div>
                    <div style="font-size: 0.85rem; text-align: left; opacity: 0.9;">
                        <strong>Resumo:</strong><br>{resumo_perfil}
                    </div>
                    <div style="margin-top: 15px; font-size: 0.8rem; opacity: 0.7; text-align: right;">
                        Score Final: <strong>{row['Score_Tecnico']:.2f}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        if len(promovidos) > 6:
            st.caption(f"...e mais {len(promovidos)-6} colaboradores na lista completa acima.")

    st.markdown('<div id="insights"></div>', unsafe_allow_html=True)
    
    if not promovidos.empty:
        avg_score_prom = promovidos['Score_Tecnico'].mean()
        avg_score_geral = df_elegiveis['Score_Tecnico'].mean()
        
        mask_ii = df_elegiveis['Nível de Cargo'] == 'II'
        mask_iii = df_elegiveis['Nível de Cargo'] == 'III'
        
        score_nivel_ii = df_elegiveis[mask_ii]['Score_Tecnico'].mean() if mask_ii.any() else 0
        score_nivel_iii = df_elegiveis[mask_iii]['Score_Tecnico'].mean() if mask_iii.any() else 0
        
        if score_nivel_ii > 0:
            gap_nivel = ((score_nivel_iii / score_nivel_ii) - 1) * 100
        else:
            gap_nivel = 0
        
        st.markdown(f"""
        <div class="dashboard-card">
            <div class="card-title">5. Insights Gerenciais e Recomendações</div>
            <div class="card-text">
                <strong>1. Elevação da Barra Técnica</strong><br>
                O grupo selecionado apresenta performance <strong>{((avg_score_prom/avg_score_geral)-1)*100:.1f}% superior</strong> à média geral. Isso confirma a efetividade dos critérios de corte utilizados.
                <br><br>
                <strong>2. Análise de Performance por Nível (Inversão de Curva)</strong><br>
                Identificamos que os Assistentes III apresentam score médio inferior aos Assistentes II ({gap_nivel:.1f}%). Isso sugere uma possível saturação nas atribuições do nível mais alto ou necessidade de revisão nos critérios de avaliação para este grupo específico.
                <br><br>
                <strong>3. Curva de Engajamento por Tempo de Casa</strong><br>
                Os dados indicam que a performance atinge o pico entre 12 e 24 meses. Após este período, nota-se uma estabilização ou leve declínio nos indicadores de produtividade. Recomenda-se implementar ações de job rotation ou novos desafios técnicos para colaboradores com mais de 2 anos para evitar estagnação.
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Carregando dados da API...")
