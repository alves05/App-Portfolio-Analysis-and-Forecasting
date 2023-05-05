import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
from   pypfopt    import risk_models
from   pypfopt    import expected_returns
import sys
import six
sys.modules['sklearn.externals.six'] = six
import mlrose
from   prophet    import Prophet

st.set_page_config(
    page_title="Active Management",
    page_icon=":bar_chart",
    initial_sidebar_state="expanded"
)
st.sidebar.image("ACTIVE-MANAGEMENT16.png", use_column_width=True)
st.sidebar.markdown('[Versão 1.0.0]')
st.title('Análise de Portfólio de Ações e Previsões com Series Temporais')
st.markdown('---')

@st.cache_data
def get_data( path ):
    # Carregando base
    data = pd.read_csv( path )
    # Convertendo coluna Date para o formato de data
    data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
    return data

def data_index( data ):
    data = data.set_index('Date')
    return data

def portfolio_optimization( data ):
    # Função fitness
    def sharpe_ratio(solucao):
        # Criando copia da base
        dataset = data.copy()
        # Normalizando os pesos
        pesos = solucao / solucao.sum()
        # Calculo do retorno medio
        returns = dataset.pct_change().mean() * 252
        # Calculo do retorno do portfolio
        portfolio_return = np.dot(returns, pesos)
        # Matriz de covariancia do portfolio
        cov_matrix = dataset.pct_change().cov()
        # Risco do portfolio
        portfolio_std_dev = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix * 252, pesos)))
        # Taxa selic do período
        risk_free = np.array([6.43, 5.77, 2.55, 5.02, 12.83]).mean() / 100
        # Calculo do indice sharpe
        sharpe_ratio = np.around((portfolio_return - risk_free) / portfolio_std_dev, 3)
        return sharpe_ratio

    # Função perosonalizada
    fitness = mlrose.CustomFitness(sharpe_ratio)
    # Definindo problema de maximização
    problema_maximizacao = mlrose.ContinuousOpt(length=len(data.columns),
                                                fitness_fn=fitness,
                                                maximize=True,
                                                min_val=0,
                                                max_val=1)
    # Executando o modelo
    melhor_solucao, melhor_custo = mlrose.genetic_alg(problema_maximizacao,
                                                      random_state=1)
    # Normalizando a melhor solucao de distribuição dos pesos
    melhor_solucao = melhor_solucao / melhor_solucao.sum()
    # Distribuição dos pesos por ação
    acoes_pesos = data.columns
    acoes_pesos = pd.DataFrame(data={'Ações': acoes_pesos, 'Pesos': melhor_solucao})
    return melhor_solucao, melhor_custo, acoes_pesos

def return_volatlity(tx_retorno, pesos):
    # Taxa de retorno anualizada
    retorno = np.around((np.sum(tx_retorno.mean() * pesos) * 252) * 100, 2)

    # Calculo da volatilidade
    volatilidade = np.around((np.sqrt(np.dot(pesos, np.dot(tx_retorno.cov() * 252, pesos)))) * 100, 2)
    return retorno, volatilidade

def income_simulation( data , pesos , capital ):
    # Criando copia da base
    data_original = data.copy()

    # Normalizando a base
    data_original = data_original / data_original.iloc[0]

    # Obtendo o rendimentos por ação e por dia
    for i, j in enumerate(data_original.columns):
        data_original[j] = data_original[j] * pesos[i] * capital

    # Agrupando em uma coluna
    data_original['Rendimentos'] = data_original.sum(axis=1, numeric_only=True)

    # Criando coluna e calculando a Taxa de retorno
    data_original['Taxa_Retorno'] = 0.0
    for i in range(1, len(data_original)):
        data_original['Taxa_Retorno'][i] = ((data_original['Rendimentos'][i] / data_original['Rendimentos'][
            i - 1]) - 1) * 100

    # Retorno em reais
    rendimento = np.around(data_original['Rendimentos'][-1] - data_original['Rendimentos'][0], 2)

    # Retorno do investimento
    roi = np.around(
        ((data_original['Rendimentos'][-1] - data_original['Rendimentos'][0]) / data_original['Rendimentos'][0]) * 100,3)

    return data_original, rendimento, roi

def stock_price_forecast( data , tick , n_forecast ):
    # Criando copia da base de dados
    dataset = data.copy()

    # Selecionando os dados da base
    dataset = dataset[['Date', tick]]

    # Renomando colunas para o padrão do algoritimo
    dataset.columns = ['ds', 'y']

    # Criando base de teste
    base_teste = dataset[len(dataset) - n_forecast:]

    # Criando, treinando e fazendo as previsões
    model = Prophet()
    model.fit(dataset)
    future = model.make_future_dataframe(periods=n_forecast)
    forecast = model.predict(future)

    # Criando base de previsões
    data_forecast = forecast[['ds', 'yhat']].tail(n_forecast)
    data_forecast.columns = ['Date', tick]

    return data_forecast

def forecast_data( data , n_forecast ):
    # Base de previsões e consolidada
    previsao_acoes = pd.DataFrame(columns=['Date'] + list(data.columns[1:]))
    for i in data.columns[1:]:
        d = stock_price_forecast(data, i, n_forecast)
        previsao_acoes['Date'] = d.iloc[:,0]
        previsao_acoes[i] = d.iloc[:,1]
        
    return previsao_acoes

def data_filter_analysis( data ):
    # definindo max e min para o filtro de datas
    min_date = datetime.strptime(str(data['Date'].min().date()), '%Y-%m-%d')
    max_date = datetime.strptime(str(data['Date'].max().date()), '%Y-%m-%d')
    # slider de datas
    st.sidebar.header('Filtros Análise')
    st.markdown('#')
    slider_value = st.sidebar.slider(
        key=1,
        label='Selecione o período',
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
    # Filtro de data
    st.session_state['Selecione o período'] = slider_value
    start_date = st.session_state['Selecione o período'][0].strftime('%Y-%m-%d')
    end_date = st.session_state['Selecione o período'][1].strftime('%Y-%m-%d')
    filtered_data = data.loc[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
    filtered_data['Date'] = filtered_data['Date'].dt.date
    filtered_data = filtered_data.set_index('Date')

    return filtered_data

def analysis_portfolio( data ):
    # Analise financeira
    retorno_diario = data.pct_change().fillna(0)
    retorno_anualizado = expected_returns.mean_historical_return(data, compounding=False, frequency=252)
    matriz_cov = risk_models.risk_matrix(data, method='sample_cov')

    return retorno_diario, retorno_anualizado, matriz_cov

def input_capital_analysis():
    # input de capital
    st.sidebar.markdown('#### Insira o capital para simulação')
    capital = st.sidebar.number_input(key=3,
                            label='mínimo de R$ 100,00:',
                            min_value=100.00,
                            format='%.2f')
    return capital

def overview_analysis( data , capital ):
    st.header('Analise dos Dados Históricos do Portfólio')
    st.markdown('#')  

    # Indicadores do portfólio
    sharpe = float(np.around(melhor_custo, 3))
    retorno = retorno_esperado
    volatilidade = volatilidade_esperada

    # Table de resultados da análise
    tabela_resultado = pd.DataFrame({'Retorno %':np.around(retorno_anualizado*100,2),
                                    'Volatilidade %':np.around(retorno_diario.std()*100,2),
                                    'Pesos %':np.around(melhor_solucao*100,2)})
    tabela_resultado.rename_axis('Ações', inplace=True)

    h1, h2 = st.columns(2)
    h1.markdown('##### Avaliação do Portfólio')
    h1.metric(label = '**Retorno %:**', value = retorno)
    h1.metric(label = '**Volatilidade %:**', value = volatilidade)
    h1.metric(label = '**Índice Sharpe:**', value = sharpe)
    h2.markdown('##### Tabela de Análises Por Ação')
    h2.dataframe(tabela_resultado)

    # multiselect ação
    acoes = data.columns[1:].tolist()
    opcao = st.multiselect(
        key=2,
        label='Selecione a(s) acão(ões)',
        options=acoes,
        default=acoes
    )

    # Gráfico de cotações
    fg = px.line(title='Gráfico de Cotações das Ações')
    for i in opcao:
        fg.add_scatter(x=filtered_data.index, y=filtered_data[i], name=i)
    fg.update_layout(yaxis=dict(gridcolor='#444444', gridwidth=0.5, zeroline=False))
    fg.update_layout(height=350, width=700)
    st.plotly_chart(fg)

    # Análise retorno do portfólio
    st.markdown("### Analise de Rendimentos do Portfólio ")
    st.markdown('#')   
    
    # Excluindo as duas ultimas colunas
    simulacao_historico = simulacao[simulacao.columns[:-2]]

    # Formatação
    capital_format = 'R$ {:,.2f}'.format(capital)
    rendimento_format = 'R$ {:,.2f}'.format(rendimento)
    roi_format = '{:,.2f} %'.format(roi)

    # Tabela para análise
    retorno_portfolio = pd.DataFrame(columns=['Ação', 'Capital $', 'Retorno $', 'Rendimento $', 'ROI %'])
    retorno_portfolio['Ação'] = filtered_data.columns
    retorno_portfolio['Capital $'] = np.around(simulacao_historico.iloc[0].values,2)
    retorno_portfolio['Retorno $'] = np.around(simulacao_historico.iloc[-1].values,2)
    retorno_portfolio['Rendimento $'] = retorno_portfolio['Retorno $'] - retorno_portfolio['Capital $']
    retorno_portfolio['ROI %'] = np.around((retorno_portfolio['Rendimento $'] / retorno_portfolio['Capital $']) * 100,2)
    retorno_portfolio = retorno_portfolio.set_index('Ação')

    s1, s2 = st.columns(2)
    s1.markdown('###### Resultados do Portfólio')
    s1.metric(label = 'Capital Aplicado:', value = capital_format)
    s1.metric(label = 'ROI:', value = roi_format)
    s1.metric(label = 'Rendimentos:', value = rendimento_format)
    s2.markdown('###### Tabela de Resultados Por Ação')
    s2.dataframe(retorno_portfolio)    

    # Gráfico da análise
    figure_simuacao = px.area(simulacao, x=simulacao.index,y=simulacao['Rendimentos'],
                            color_discrete_sequence=['#a69500'], title='Rendimentos Últimos 5 anos')
    figure_simuacao.update_traces(fill='tozeroy', fillcolor='rgba(255, 191, 0, 0.05)')
    figure_simuacao.update_layout(yaxis=dict(gridcolor='#444444', gridwidth=0.5, zeroline=False))
    figure_simuacao.update_layout(height=400, width=700)

    st.plotly_chart(figure_simuacao)

    return None

def data_filter_forecast( data ):
    # Convertendo coluna Date para o formato de data
    data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")

    # definindo datas max e min
    min_date_prev = datetime.strptime(str(data['Date'].min().date()), '%Y-%m-%d')
    max_date_prev = datetime.strptime(str(data['Date'].max().date()), '%Y-%m-%d')

    # Filtro slide tempo
    st.sidebar.header('Filtros Simulação')
    slider_value_prev = st.sidebar.slider(
        key=4,
        label='Selecione o período',
        min_value=min_date_prev,
        max_value=max_date_prev,
        value=(min_date_prev, max_date_prev)
    )
    # aplicando filtro na base
    st.session_state['Selecione o período'] = slider_value_prev
    start_date_prev = st.session_state['Selecione o período'][0].strftime('%Y-%m-%d')
    end_date_prev = st.session_state['Selecione o período'][1].strftime('%Y-%m-%d')
    filtered_data_prev = data.loc[(data['Date'] >= start_date_prev) & (data['Date'] <= end_date_prev)]
    filtered_data_prev['Date'] = filtered_data_prev['Date'].dt.date
    filtered_data_prev = filtered_data_prev.set_index('Date')

    return filtered_data_prev

def input_capital_forecast():
    # input do valor do capital
    st.sidebar.markdown('#### Insira o capital para simulação')
    capital_prev = st.sidebar.number_input(key=5,
                                label='mínimo de 100,00:',
                                min_value=100.00,
                                format='%.2f')
    return capital_prev

def overview_forecast( capital_prev , simulacoes_previsoes , rendimento_prev , roi_prev ):
    st.markdown('---')
    st.header('Simulação de Rendimentos Futuros')
    st.caption('Rendimentos em 365 dias')
    st.markdown('#')

    simulacao_previsoes_acoes = simulacoes_previsoes[simulacoes_previsoes.columns[:-2]]

    # Formatação
    capital_prev_format = 'R$ {:,.2f}'.format(capital_prev)
    rendimento_format_prev = 'R$ {:,.2f}'.format(rendimento_prev)
    roi_format_prev = '{:,.2f} %'.format(roi_prev)

    # Table de análise da simulação
    retorno_previsao = pd.DataFrame(columns=['Ação', 'Capital $', 'Retorno $', 'Rendimento $', 'ROI %'])
    retorno_previsao['Ação'] = simulacao_previsoes_acoes.columns
    retorno_previsao['Capital $'] = np.around(simulacao_previsoes_acoes.iloc[0].values,2)
    retorno_previsao['Retorno $'] = np.around(simulacao_previsoes_acoes.iloc[-1].values,2)
    retorno_previsao['Rendimento $'] = retorno_previsao['Retorno $'] - retorno_previsao['Capital $']
    retorno_previsao['ROI %'] = np.around((retorno_previsao['Rendimento $'] / retorno_previsao['Capital $']) * 100,2)
    retorno_previsao = retorno_previsao.set_index('Ação')

    x1, x2 = st.columns(2)
    x1.markdown('###### Resultados Esperados')
    x1.metric(label = 'Capital Aplicado:', value = capital_prev_format)
    x1.metric(label = 'ROI:', value = roi_format_prev)
    x1.metric(label = 'Rendimentos:', value = rendimento_format_prev)
    x2.markdown('###### Tabela de Resultados Esperados Por Ação')
    x2.dataframe(retorno_previsao)

    # Gráfico da simualção
    figure_projecao = px.line(simulacoes_previsoes, x=simulacoes_previsoes.index,
                            y = simulacoes_previsoes['Rendimentos'],color_discrete_sequence=['#a69500'],
                            title="Rendimentos Futuros")
    figure_projecao.update_layout(yaxis=dict(gridcolor='#444444', gridwidth=0.5, zeroline=False))
    figure_projecao.update_layout(height=400, width=700)

    st.plotly_chart(figure_projecao)
    st.markdown('---')

    return None

if __name__ == "__main__":

    path = 'portfolio.csv'

    df = get_data( path )

    filtered_data = data_filter_analysis(df)

    retorno_diario, retorno_anualizado, matriz_cov = analysis_portfolio( filtered_data )

    melhor_solucao, melhor_custo, _ = portfolio_optimization(filtered_data)

    retorno_esperado, volatilidade_esperada = return_volatlity(retorno_diario, melhor_solucao)

    capital = input_capital_analysis()

    simulacao, rendimento, roi = income_simulation(filtered_data, melhor_solucao, capital)
 
    dataset_previsao = forecast_data( df , 365)
    
    filtered_data_prev = data_filter_forecast(dataset_previsao)

    data = data_index(df)

    solucao, _, _ = portfolio_optimization(data)

    capital_prev = input_capital_forecast()

    simulacoes_previsoes, rendimento_previsoes, roi_previsoes = income_simulation(filtered_data_prev, solucao, capital_prev)

    overview_analysis( df, capital )

    overview_forecast( capital_prev, simulacoes_previsoes, rendimento_previsoes, roi_previsoes )