# Por Adolfo García, Rodrigo Guerrero y Nicolás Sepulveda
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import nltk  
import re   
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import preprocessor as p
from sklearn.feature_extraction.text import CountVectorizer

def preproceso_tweet(row):
    """
    Esta función tiene limpiar los tweets del dataframe, quitando URL, hashtags, menciones por filas.
    Parametros:
        'row': filas 
    """    
    text = row['content']
    text = p.clean(text)
    return text

def preproceso_palabras(df):
    """
    Esta función tiene limpiar los tweets del dataframe, quitando puntos, comas, simbolos exclamación, interrogación y pasa mayusculas a minusculas. Además quita stopwords, conectores comunes del lenguaje ingles.
     
    """
    remove_rt = lambda x: re.sub("^\d+\s|\s\d+\s|\s\d+$", " ",x)
    rt = lambda x: re.sub("[^\w\s]", " ",x)
    df['content_recod'] = df['content_recod'].map(remove_rt).map(rt)
    df['content_recod'] = df['content_recod'].str.lower()
    stop = stopwords.words('english')
    df['content_recod'] = df['content_recod'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))




def cuenta_nan(df):
    """
    Esta función tiene como objetivo contabilizad la cantidad de NaN,
donde el input es el df y el output es la contabilización de dichos valores
    Parametros:
        'df': dataframe
    """
    for i in df.columns:
        if True in df[i].isna().value_counts():
            print(f'\033[1mLa variable {i} tiene {df[i].isna().value_counts().get(True)} valores perdidos\033[0m')
        else:
            print(f'La variable {i} no posee valores perdidos')
    
   
def loop_histogramas(df):
    """
    
    Esta función tiene como objetivo generar cierta cantidad de histogramas tanto como sean la cantidad de variables dentro del dataframe  l
donde el input es el df y el output serán gráficos de cada variable
    Parametros:
        'df': dataframe, de la cual se extraerán las variables

    
    """
    
    variables = df.columns

    a = len(variables) 
    b = 1  
    c = 1  

    fig = plt.figure(figsize=(10,60))

    for i in variables:
        plt.subplot(a, b, c)
        plt.title(f'{i}')
        plt.xlabel(i)
        sns.histplot(df[i])
        c = c + 1

    plt.tight_layout()
    plt.show()
 
def binarizador_op(df):
    """
    
    Esta función tiene como objetivo binarizar variables categoricas independientes del numero de categorias que ésta tenga, 
    haciendo uso de drop_first=True para eliminar primera columna, a modo no de ser redundante en la información.
    Retorna el dataframe con variables categóricas binarizadas y sin las columnas utilizadas para la binarización.
    
    Parametros:
        'df': dataframe, de la cual se extraerán las variables

    
    """
    
    variables_iterar = []
    variables_dropear = []
    
    for k in df.columns:
        if df[k].dtype == 'O':
            variables_iterar.append(k)
            variables_dropear.append(k)

    for i in variables_iterar:
        for j in df[i].unique()[1:]:
            df[f'{i}_' + j] = np.where(df[i] == j, 1, 0)


    df = df.drop(variables_dropear, inplace=True, axis=1)
    return df
        
        
def graficaHist(df, var):
    """
    
    Esta función tiene como objetivo generar un gráfico de histograma con la variable especifica que se desea visualizar,donde el input es el df y la variable y el output será el gráfico de la variable analizada 
    Parametros:
        'df': dataframe, de la cual se extraerán las variables
        'var': variable que se desea analizar
    
    """
    
    plt.title(f'Variable: {var}') 
    sns.histplot(df[var])
    
def resumen_metricas_bondad(mod):
    """
    
    Esta función tiene como objetivo entregar una tabla resumen de los principales parametros de bondad de ajuste de la regresión. Hace uso de .summary2() del modelo de regresión generado por statsmodels.
    Parametros:
        'modelo': El modelo generado.
    
    """
    fit = pd.DataFrame({'Statistics': mod.summary2().tables[0][2][:], 'Value': mod.summary2().tables[0][3][:]})
    print("\nParametros bondad de ajuste\n", fit)  
    
def variables_significativas_t(mod, show=False):
    """
    
    Esta función tiene como objetivo entregar las variables con un P-value menor a 0.05, de acuerdo a una distribucion t-student.
    Parametros:
        'mod': El modelo generado.
        'show': Por defecto False, si es True, mostrará la lista de variables significativas.
    
    """

    lista_pvalue = list(mod.summary2().tables[1]['P>|t|'].values)
    variables = list(mod.summary2().tables[1]['P>|t|'].index)
    diccionario = dict(zip(variables, lista_pvalue))
    del diccionario['Intercept']

    borrar= []
    variables_significativas = []
    for i in diccionario:
        if diccionario[i] < 0.05:
            variables_significativas.append(i)
    if show == True:
        print(variables_significativas)

    variables_sumadas = "+".join(variables_significativas)
    return variables_sumadas

def variables_significativas_z(mod, show=False):
    """
    
    Esta función tiene como objetivo entregar las variables con un P-value menor a 0.05, de acuerdo a una distribucion z.
    Parametros:
        'mod': El modelo generado.
        'show': Por defecto False, si es True, mostrará la lista de variables significativas.
    
    """

    lista_pvalue = list(mod.summary2().tables[1]['P>|z|'].values)
    variables = list(mod.summary2().tables[1]['P>|z|'].index)
    diccionario = dict(zip(variables, lista_pvalue))
    del diccionario['Intercept']

    borrar= []
    variables_significativas = []
    for i in diccionario:
        if diccionario[i] < 0.05:
            variables_significativas.append(i)
    if show == True:
        print(variables_significativas)

    variables_sumadas = "+".join(variables_significativas)
    return variables_sumadas

 

    
    
    
def palabras_por_genero(df, genero):
    """
    
    Esta función tiene como objetivo entregar el conteo de las palabras de nuestro dataframe por clase, es decir si es positivo o negativo 
    Parametros:
        'df': Dataframe suministrado por el cliente.
        'genero': Es el "sentiment" asociado del que se realizará la busqueda, se utilizarán positivo o negativo.
    
    """
    
    # instanciamos un objeto
    count_vectorizer=CountVectorizer(stop_words='english')
    # Implementamos los pasos fit y transform
    count_vectorizer_fit = count_vectorizer.fit_transform(df[df['sentiment']==genero]['content'])
    # Extraemos tokens (palabras)
    words = count_vectorizer.get_feature_names()
    # extraemos las frecuencias por palabras
    words_freq = count_vectorizer_fit.toarray().sum(axis=0)

    #creamos un df para la matrix de palabras por género
    df_palabras = pd.DataFrame({'Palabra': words,'Frecuencia': words_freq })

    #obtenemos un df ordenado para las primeras 100 palabras mas frecuentes
    df_palabras_mas_frecuentes = df_palabras.sort_values(by=['Frecuencia'], ascending=False).head(100)

    #retornamos el df ordenado
    return df_palabras_mas_frecuentes    
    
    
    
#creamos una función para graficar la frecuencia de las palabras 
def graficos_por_genero(df):
    """
    
    Utiliza el dataframe para generar el gráfico con el conteo de palabras por cada sentimiento
        'df': Dataframe suministrado por el cliente.
    
    """    
    
    plt.figure(figsize=(18, 25))
    generos = df['sentiment'].unique()
    for n, i in enumerate(generos):
        df_palabras_mas_frecuentes = palabras_por_genero(df, i)

        # generamos tantos subplots en una fila, como géneros existan
        plt.subplot(1, len(generos), n + 1)

        #creamos el gráfico para el género
        ax = sns.barplot(y='Palabra',
                         x='Frecuencia',
                         data=df_palabras_mas_frecuentes)
        plt.title(f'Frecuencias por palabra para {i}')
        #agregamos los totales al final de la barra
        for p in ax.patches:
            total = f'{int(p.get_width()):,}'.replace(',','.')
            x = p.get_x() + p.get_width() + 0.06
            y = p.get_y() + p.get_height()/2
            ax.annotate(total, (x, y))
    
    
    
    








