import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from PIL import Image

st.set_option('deprecation.showPyplotGlobalUse', False)

DATA = pd.read_excel("stats_new.xlsx", sheet_name="DATA")
    # BORRAMOS LAS COLUMNAS QUE YA NO HACEN FALTA
DATA = DATA.drop(["H","HR","%BUENA ZONA","%MALA ZONA"], axis = 1)
DATA1 = DATA[DATA['ERA']<=8]
logo = Image.open(r'imagenes/logo.jpg')
st.sidebar.image(logo, width=100)
st.sidebar.header("Lanzadores MLB 2022")
st.sidebar.write(" ")
st.sidebar.write(" ")
option = st.sidebar.selectbox(
    'Selecciona una pagina para navegar por la app',
    ('Presentacion','Data de los lanzadores', 'Graficas',"Top lanzadores por K y ERA","Top Mejores lanzadores","Correlacion","Modelo de regresion"))
if option == 'Presentacion':
    st.write(" ")
    st.write(" ")
    stats = Image.open(r'imagenes/STATS.jpg')
    st.sidebar.header('Recursos utilizados')
    st.sidebar.markdown('''
- [Pagina MLB](https://www.mlb.com/stats/pitching/avg-allowed-by-the-pitcher/2022?sortState=ascs) de donde se extrajo la data
- [Cuaderno de trabajo en Google colab](https://colab.research.google.com/drive/1pk54HRYvUookOAm2m7BCdafkBHxOaYnX?usp=sharing) donde se realizo limpieza y tratamiento de la data
''')
    col1, col2, col3 = st.columns((1,4,1))
    col2.image(stats, width=300)
    st.write(" ")
    st.write(" ")
    st.markdown("# Analisis de las estadisticas de los lanzadores de la MLB 2022")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    st.write(" ")
    col1, col2, col3 = st.columns(3)
    col1.expander("Presentado por").write("Gustavo Boada")
    col3.expander("Git-hub").write("[Repositorio](https://github.com/gboada23/Analisis-MLB)")
    col2.expander("Contacto").write("""
    [Perfil Linkedln](https://www.linkedin.com/in/gboada23/)

    [Portafolio](https://portafolio-gustavo-boada.streamlit.app/)""")

elif option == 'Data de los lanzadores':
    DATA = pd.read_excel("stats_new.xlsx", sheet_name="DATA")
    # BORRAMOS LAS COLUMNAS QUE YA NO HACEN FALTA
    DATA = DATA.drop(["H","HR","%BUENA ZONA","%MALA ZONA"], axis = 1)



    st.header("Data de los lanzadores interactiva:")
    st.markdown("#### Puedes seleccionar las estadisticas que desees")
    # Creamos una lista con los nombres de las columnas del dataframe
    columnas = list(DATA.columns)

    # Creamos un multiselect para seleccionar las columnas a mostrar
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    columnas_seleccionadas = st.sidebar.multiselect('Selecciona las columnas a mostrar', columnas, default=["NOMBRE","APELLIDO"])

    # Mostramos el dataframe con las columnas seleccionadas
    data_filt = DATA[columnas_seleccionadas]
    st.dataframe(data_filt,width=550, height=400)
    
    descargado = False

    def descargar_excel1():
        with pd.ExcelWriter('data_filtrada.xlsx') as writer:
            data_filt.to_excel(writer, index=False)
        with open('data_filtrada.xlsx', 'rb') as f:
            bytes_data = f.read()
        st.sidebar.download_button(label="Descargar", data=bytes_data, file_name='data_filtrada.xlsx', mime='application/vnd.ms-excel')
       
    st.sidebar.write(" ")
    st.sidebar.write(" ")
    st.sidebar.write('Para descargar la data filtrada como Excel, presiona el siguiente botón:')
    descargar_excel1()

    st.expander("INFO").write("Esta tabla pertenece a la data limpia y tratada a travez de Google colab con todos los lanzadores con estadisticas no nulas el archivo se llama [Pitcher MLB 2022.ipynb](https://colab.research.google.com/drive/1pk54HRYvUookOAm2m7BCdafkBHxOaYnX?usp=sharing)")


elif option == 'Graficas':
    st.title('Graficas')
# Contamos el número de lanzadores zurdos y derechos
    conteo_brazos = DATA['BRAZO'].value_counts()
    # Creamos una lista de colores para usar en el gráfico
    colores = ['#00008B','#ADD8E6']
    # Creamos el gráfico de anillos con Plotly
    # Creamos el gráfico de anillos con Plotly
    fig = go.Figure(data=[go.Pie(labels=["Derechos","Zurdos"], 
                                values=conteo_brazos.values, 
                                hole=.4, 
                                marker=dict(colors=colores),
                                textinfo='label+percent')])
    # Agregamos un título centrado sobre el gráfico
    fig.update_layout(title={
            'text': "Porcentaje de lanzadores por tipo de brazo",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font=dict(size=20))
    # Mostramos el gráfico en Streamlit
    st.plotly_chart(fig)
    expandir = st.expander("Ver interpretacion")
    expandir.write("Podemos observar en la grafica que de los 490 lanzadores el 73.5% son derechos y el resto lanzadores zurdos")
    
    st.subheader("Ahora veremos algo curioso en esta pequeña tabla")
    # tabla de lanzadores zurdos y derechos
    k_mean = DATA.groupby('BRAZO')['K'].mean()
    bb_mean = DATA.groupby('BRAZO')['BB'].mean()
    st.dataframe(pd.DataFrame({'K': k_mean, 'BB': bb_mean}),width=700, height=100)
    expander = st.expander("Ver interpretacion")
    expander.write("""Esta tabla no nos dice mucho con respecto a los K 
            solamente podemos decir que en promedio los lanzadores Zurdos 
            tienden a dar mas bases por bolas que los derechos y es muy
            curioso porque segun la grafica de arriba son muchos mas lanzadores
            derechos que zurdos, entonces podemos asumir que si los lanzadores
            zurdos igualaran a la cantidad de lanzadores derechos fueran los 
            lanzadores con muchos BB duplicando el promedio de los lanzadores Derechos""")

    # Crear una figura de barras en Plotly
    fig = go.Figure()

    # Filtrar los datos para obtener los lanzadores zurdos y derechos

    DERECHOS = DATA[DATA["BRAZO"] == "R"]
    ZURDOS = DATA[DATA["BRAZO"] == "L"]

    fig.add_trace(go.Bar(
        x=['K', 'BB', 'MPH'],
        y=ZURDOS[['K', 'BB', 'MPH']].mean(),
        name='Zurdos',
        error_y=dict(type='data', array=ZURDOS[['K', 'BB', 'MPH']].std()),
        width=0.4
    ))
    fig.add_trace(go.Bar(
        x=['K', 'BB', 'MPH'],
        y=DERECHOS[['K', 'BB', 'MPH']].mean(),
        name='Derechos',
        error_y=dict(type='data', array=DERECHOS[['K', 'BB', 'MPH']].std()),
        width=0.4
    ))
    # Personalizar la figura
    fig.update_layout(
        xaxis=dict(tickmode='array', tickvals=[0, 1, 2], ticktext=['K', 'BB', 'MPH']),
        yaxis=dict(title='Valor promedio'),
        legend=dict(title='Brazo'),
        barmode='group'
    )
    # Mostrar la figura en Streamlit
    st.plotly_chart(fig)
    analisis = """En la grafica de arriba podemos observar que no existe gran diferencia en promedio de las 
            estadisticas de un lanzador zurdo o derecho sin embargo los lanzadores zurdos en los 
            Boletos segun la linea negra indica la variabilidad de los datos en cada grupo de la 
            variable categórica por lo que es mayor la variabilidad en lanzadores zurdos por lo que 
            podemos alegar que son lanzadores mas descontrolados cosa que en la tabla pequeña de arriba ya se notaba"""
    expander1 = st.expander("Ver interpretacion")
    expander1.write(analisis)

    fig = px.box(DATA, x='BRAZO', y='ERA', points=False, title='Boxplot de Efectividad por Brazo')
    fig.update_layout(xaxis_title='Brazo', yaxis_title='Efectividad')
    st.plotly_chart(fig)
    analisis = """Segun la grafica de caja no hay gran diferencia entre si un lanzador
            en zurdo o derecho en la EFECTIVIDAD solamente vemos que los 
            lanzadores Derechos tiene mas datos alejados del promedio por lo que la 
            variabildad es mas alta que la de los zurdos"""
    expander2 = st.expander("Ver interpretacion")
    expander2.write(analisis)

elif option == "Top lanzadores por K y ERA":
    st.title('Mejores lanzadores en K y ERA')
    columnas = list(DATA.columns)

    entrada = st.sidebar.slider('Cantidad de lanzadores a mostrar en la tabla', 1, 100, 10)

    # Creamos un multiselect para seleccionar las columnas a mostrar
    col1, col2 = st.columns(2)

    # Columna 1
    col1.subheader(f"TOP {entrada} lanzadores con mas ponches")
    columnas_seleccionadas1 = st.sidebar.multiselect('Si quieres mostrar otra estadistisca a parte de los K en la primera tabla selecciona aqui', columnas, default=["NOMBRE","APELLIDO","K"])
    top_k = DATA[columnas_seleccionadas1]
    top_k = top_k.nlargest(entrada, 'K')
    col1.dataframe(top_k,width=500, height=400)

    # Botón de descarga para Columna 1
    def descargar_excel2(top_k):
        with pd.ExcelWriter('data.xlsx') as writer:
            top_k.to_excel(writer, index=False)
        with open('data.xlsx', 'rb') as f:
            bytes_data = f.read()
        col1.download_button(label="Descargar", data=bytes_data, file_name='Lanzadores_ponchadores.xlsx', mime='application/vnd.ms-excel', key='descargar_excel_k')

    col1.write('Para descargar la data filtrada como Excel, presiona el siguiente botón:')
    descargar_excel2(top_k)

    # Columna 2
    col2.subheader(f"TOP {entrada} lanzadores con mejor efectividad ")
    columnas_seleccionadas2 = st.sidebar.multiselect('Si quieres mostrar otra estadistisca a parte de la ERA en la segunda tabla selecciona aqui', columnas, default=["NOMBRE","APELLIDO","ERA"])
    DATA4 = DATA[DATA['IL']> 100]
    top_era = DATA4.sort_values("ERA", ascending= True)
    top_era = top_era[columnas_seleccionadas2]
    top_era = top_era.head(entrada)
    col2.dataframe(top_era,width=500, height=400)

    # Botón de descarga para Columna 2
    def descargar_excel3(top_era):
        with pd.ExcelWriter('data2.xlsx') as writer:
            top_era.to_excel(writer, index=False)
        with open('data2.xlsx', 'rb') as f:
            bytes_data = f.read()
        col2.download_button(label="Descargar", data=bytes_data, file_name='Lanzadores_efectivos.xlsx', mime='application/vnd.ms-excel', key='descargar_excel_era')

    col2.write('Para descargar la data filtrada como Excel, presiona el siguiente botón:')
    descargar_excel3(top_era)

elif option == "Top Mejores lanzadores":
    
    st.title('Mejores lanzadores MLB 2022')
    st.write(" ")
    expandir = st.expander("Desplegar explicacion")
    expandir.write(f""" Desde mi punto de vista obtendre los mejores lanzadores utilizando un enfoque de ranking en el que le doy un peso las estadisticas que considero mas importante para que alguien sea considerado uno de los mejores lanzadores de una temporada
    En este caso, voy utilizar las siguientes estadísticas:
    K, BB, ERA y WHIP

    Para darle peso a las estadísticas, podemos estandarizar cada una de ellas restandoles la media y dividiendo entre su desviacion tipica y luego multiplicarlas por un coeficiente que represente su importancia relativa y sumamos las puntuaciones para cada lanzador.

    Aqui veremos los scalares que voy a usar
    *   K: 0.3
    *   BB: -0.1
    *   ERA: -0.5
    *   whip: -0.1
    
    Si se dan cuenta 4 de los 3 escalares le doy puntuacion negativa porque lo que queremos lograr es minimizar las BB (Bases por bolas), Minimzar la ERA (Efectividad), y minimizar el Whip que me indica el porcentajes de embasados por innings lanzados, mientras que los K son positivos porque queremos maximixar dicha estadistisca.""")
    col1, col2, col3 = st.columns(3)
    best_players = Image.open(r'imagenes/best_pitchers.jpg')
    st.image(best_players, width=620)
    entrada1 = st.sidebar.slider('Cantidad de los Mejores lanzadores que muestra la tabla', 1, 100, 10)
  
    # coeficientes
    coef_k = 0.3
    coef_bb = -0.1
    coef_era = -0.5
    coef_whip = -0.1
    # Normalizar cada estadística y calcular la puntuación para cada lanzador
    DATA1['K_norm'] = (DATA1['K'] - DATA1['K'].mean()) / DATA1['K'].std()
    DATA1['BB_norm'] = (DATA1['BB'] - DATA1['BB'].mean()) / DATA1['BB'].std()
    DATA1['ERA_norm'] = (DATA1['ERA'] - DATA1['ERA'].mean()) / DATA1['ERA'].std()
    DATA1['AVG_norm'] = (DATA1['AVG'] - DATA1['AVG'].mean()) / DATA1['AVG'].std()
    DATA1["WHIP_norm"] = (DATA1['WHIP'] - DATA1['WHIP'].mean()) / DATA1['WHIP'].std()
    DATA1['score'] = coef_k * DATA1['K_norm'] + coef_bb * DATA1['BB_norm'] + coef_era * DATA1['ERA_norm'] + coef_whip * DATA1["WHIP_norm"]
    col_numericas1 = ["NOMBRE","APELLIDO","IL", "K", "BB","EDAD","%K","%BB","AVG","ERA","MPH","WHIP","score"]
    # Ordenar los lanzadores por puntuación y seleccionar los primeros 10 abridores con mas de 120 IL
    DATA3 = DATA1[DATA1["IL"]> 120 ]
    columnas_seleccionadas4 = st.sidebar.multiselect('Si quieres mostrar algunas estadistiscas selecciona aqui', col_numericas1, default=["NOMBRE","APELLIDO","K","BB","WHIP","ERA","score"])
    top_10_pitchers = DATA3.nlargest(entrada1, 'score')
    top_10_pitchers = top_10_pitchers[columnas_seleccionadas4]

    st.dataframe(top_10_pitchers,width=620, height=400)
    def descargar_excel4():
        with pd.ExcelWriter('mejores_lanzadores.xlsx') as writer:
            top_10_pitchers.to_excel(writer, index=False)
        with open('mejores_lanzadores.xlsx', 'rb') as f:
            bytes_data = f.read()
        st.sidebar.download_button(label="Descargar", data=bytes_data, file_name='mejores_lanzadores.xlsx', mime='application/vnd.ms-excel')

    st.sidebar.write('Para descargar la datade los mejores lanzadores como Excel, presiona el siguiente botón:')
    descargar_excel4()
    a = st.expander("NOTA")   
    a.write("Podemos observar que de los 10 mejores lanzadores por ranking que obtuve 2 ganaron el CY young y 2 quedaron finalistas a dicho premio")
elif option == "Correlacion":
    st.title('Matriz de correlacion y grafica de dispersion')

    st.subheader("Correlacion de las variables numericas")
    col_numericas = ["IL", "K", "BB","EDAD","%K","%BB","AVG","ERA","MPH","WHIP"]
    columnas_corr = st.sidebar.multiselect('Selecciona las columnas que se pueden correlacionar', col_numericas, default=["AVG","ERA"])

    # Creación de matriz de correlación
    if columnas_corr:
        data_corr = DATA[columnas_corr].corr()
        # Gráfico de heatmap
        st.write('Matriz de correlación interactiva')
        fig, ax = plt.subplots(figsize=(5,4))
        ax.set_facecolor('black')
        sns.heatmap(data_corr, annot=True)
        st.pyplot(fig)
        st.write("Seleccione la correlacion entre AVG y ERA para graficarla y verificar que sigue una correlacion lineal")

    # grafica de dispersion
    st.write(" ")
    st.subheader("Grafica de dispersion del AVG y ERA")
    fig = px.scatter(DATA, x='AVG', y='ERA')

    st.plotly_chart(fig)
    variable = """st.header("Verificamos la distribucion de las variables 'AVG' y 'ERA'")

    # Histograma de AVG 

    st.subheader('Histograma de AVG')
    data_hist = {'x': DATA1['AVG'], 'type': 'histogram', 'nbinsx': 20}
    fig = go.Figure(data=data_hist)
    st.plotly_chart(fig, height=200, width=200)

    # Histograma de ERA 
    st.subheader('Histograma de ERA')
    data_hist = {'x': DATA1['ERA'], 'type': 'histogram', 'nbinsx': 20}
    fig = go.Figure(data=data_hist)
    st.plotly_chart(fig, height=200, width=200)"""
    st.expander("Interpretacion").write("Podemos evidenciar en la grafica de dispersion que se evidencia una relacion Lineal entre el AVG y la ERA")
else:
    st.title('Modelo de regresion')
    model = LinearRegression()

    model.fit(DATA1['AVG'].values.reshape(-1, 1), DATA1['ERA'].values.reshape(-1, 1))

    # Coeficientes de la regresión
    coef = round(model.coef_[0][0],2)
    intercepto = round(model.intercept_[0],2)

    # Coeficiente de determinación (R^2)
    y_pred = model.predict(DATA1['AVG'].values.reshape(-1, 1))
    r2 = round(r2_score(DATA1['ERA'].values.reshape(-1, 1), y_pred),2)
    no_explain = round(1 - r2,2)
    # Crear widgets de entrada para valores del AVG
    AVG_entrada = st.sidebar.slider('AVG', float(DATA1['AVG'].min()), float(DATA1['AVG'].max()), float(DATA1['AVG'].mean()))
    # Calcular la predicción
    era_prediccion = model.predict([[AVG_entrada]])

    # Mostrar la predicción

    st.sidebar.markdown(f'### Para un lanzador al que le batean *{round(AVG_entrada, 2)}* se espera tener una ERA de *{round(era_prediccion[0][0],2)}*')

    # Dividir los datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(DATA1['AVG'], DATA1['ERA'], test_size=0.2, random_state=42)

    # Entrenar el modelo con los datos de entrenamiento
    model.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

    # Coeficiente de determinación con los datos de entrenamiento
    r2_train = round(model.score(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1)),1)
    # Coeficiente de determinación con los datos de test
    r2_test = round(model.score(X_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1)),1)

    # Graficar la recta de regresión y la distribución de las variables
    sns.set_style('dark')
    fig, ax = plt.subplots(figsize=(5,4))
    sns.regplot(x=DATA1['AVG'], y=DATA1['ERA'], line_kws={'color': 'red'})
    ax.set_xlabel('AVG')
    ax.set_ylabel('ERA')
    ax.set_title('Regresión lineal simple entre AVG y ERA')
    ax.set_facecolor('black')
    st.pyplot(fig)

    # Coeficientes de la regresión
    st.sidebar.write("Coeficiente de la pendiente:", coef)
    st.sidebar.write("Intercepto:", intercepto)
    st.sidebar.write('Coeficiente de determinación (R^2) con todos los datos:', r2)
    st.sidebar.write('Coeficiente de determinación (R^2) con los datos de entrenamiento:', r2_train)
    st.sidebar.write('Coeficiente de determinación (R^2) con los datos de test:', r2_test)
    st.sidebar.write('Varianza no explicada:', no_explain)
    st.subheader("Conclusion")
    st.expander("Ver conclusion").write("""Después de analizar los datos, podemos llegar a varias conclusiones:

-   Hay una correlación negativa entre la velocidad promedio de la recta (MPH) y la ERA, sin embargo la correlacion es muy debil lo que nos dice que no necesariamente un lanzador con rectas rapidas 98 MPH es mas efectivo que uno con rectas de 90 MPH.

-   Los lanzadores zurdos parecen tener un ligero beneficio en términos de efectividad, ya que su ERA promedio es un poco más baja que la de los lanzadores derechos quizas tambien tiene que ver porque los lanzadores zurdos son menos.

-   No parece haber una relación clara entre el tipo de brazo de un lanzador y su efectividad.

-   Y Por ultimo que el modelo no es lo suficientemente bueno debido a que solo recogera de manera correcta el 56% de las predicciones ya que la varianza no me explica el 43% de las predicciones, lo que quiere decir que quizas hayan otras variables que interfieran para mejorar este modelo yo tomaria en cuenta incluir mas variables significativas para lograr una regresion lineal multiple y coneguir un buen modelo que se adapte correctamente a los datos.""")
