# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:14:30 2022

@author: david
"""
#Librerias

import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy 
##NLTK
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
#Importamos las funciones vistas en clase
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
stopSpanish=set(stopwords.words('spanish'))
##split
from sklearn.model_selection import train_test_split
#base line
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import scipy.sparse


#Funciones
#Analisis palabras
def contadorP(listaTxt,corte):
    opinion_len = pd.Series([len(op.split()) for op in listaTxt])
    opinion_len[opinion_len<corte ].plot(kind='box')
    print(opinion_len.describe())
#NLTK
def limpiarNLTK(listaTxt,categoria):   
    resumen=[]
    for i in range (n_noticias):
        text_tokens = word_tokenize(listaTxt[i])
        tokens_without_sw = [word for word in text_tokens if not word in stopSpanish]
        tokens_without_sw = [word.lower() for word in tokens_without_sw if word.isalpha()]
        resumen.append(' '.join(tokens_without_sw))
    return resumen

    
    
def conexiondb(dbfile):
    try:
        con = sqlite3.connect(dbfile)    
    except Exception as e:
        print(e)
    #Now in order to read in pandas dataframe we need to know table name
    cursor = con.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    #print(f"Table Name : {cursor.fetchall()}")
    Names=cursor.fetchall()
    df=[]
    for db in Names:
        TableName=db[0]
        df.append(pd.read_sql_query('SELECT * FROM '+TableName, con))
    con.close()
    return df

# recortador de textos
def Recorte(DF,minp=100,maxp=500):
    listaTxt=DF["articulo"]
    #listaRes=DF["resumen"]
    #Elementos=[]
    opinion_len = pd.Series([len(op.split()) for op in listaTxt])
    #Elementos=listaTxt[opinion_len>size]
    
    
    #DF2 = (DF.drop(DF[opinion_len<=minp] or DF[opinion_len>maxp])).reset_index(drop=True)
    DF2 = (DF.drop(DF[opinion_len<=minp].index | DF[opinion_len>maxp].index)).reset_index(drop=True)
    #DF2 = (DF.drop(DF[opinion_len<=minp or opinion_len>maxp].index)).reset_index(drop=True)
    #opinion_len=opinion_len[opinion_len>minp or opinion_len<maxp]
    return DF2
  
##Cargando data
#dbfile = '/content/drive/MyDrive/Datos/NoticiasReforma2009.db'
#dbfile="/content/drive/Othercomputers/My Computer/Computo Estadistico/PIA-CD2/NoticiasReforma2009.db"
dbfile='D:/Computo Estadistico/PIA-CD2_v2/NoticiasReforma2009.db'
DATA=conexiondb(dbfile)
df_N2009=DATA[1]
#df_R2009=DATA[0]

##Preproceso
#Se eliminan las columnas no ocupamos
df_N2009 = df_N2009[['titulo','resumen','articulo']].copy()
n_noticias=(df_N2009).shape[0]

#contador de palabras antes de limpiar
##Resumen
contadorP(df_N2009["resumen"],50)
##Articulo
contadorP(df_N2009["articulo"],500)

##removemos todas las noticias con menos de  100 palabras
df_N2009v2=Recorte(df_N2009,minp=100,maxp=400)
contadorP(df_N2009v2["articulo"],500)




#Limpiar nltk
categoria="articulo"
resumenA=limpiarNLTK(df_N2009v2[categoria],categoria)
categoria="resumen"
resumenR=limpiarNLTK(df_N2009v2[categoria],categoria)

#contador de palabras despues de limpiar
##Resumen
contadorP(resumenR,30)
##Articulo
contadorP(resumenA,350)

#Split
X_train, X_test, y_train, y_test  = train_test_split(resumenA,resumenR,test_size=0.2, random_state=20, shuffle=True)


#!pip install --upgrade scipy networkx
#!pip install networkx==2.6.3
#!pip install scipy==1.8.1

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop()    
    return sentences

def read_article2(filedata):    
    article = filedata.split(".") #separa los textos por puntos que hay
    sentences = []

    for sentence in article:
        #print(sentence)
        sentence=sentence.replace("  ","")
        tokens=sentence.replace("[^a-zA-Z]", " ").split(" ")
        #print(tokens)
        #print(tokens)
        if len(tokens)>3:
            sentences.append(tokens)
    sentences.pop()  
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix
'''
Texto=resumenA[0]
Texto=df_N2009["articulo"][8]
'''

Texto=df_N2009v2["articulo"][0]

def generate_summary(Texto, top_n=5,c=0,MaxP=400):
    print(c)
    stop_words = stopwords.words('spanish')
    summarize_text = []
    # Step 1 - Read text anc split it
    sentences =  read_article2(Texto) #recibe texto por texto
    
    
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph,  max_iter=MaxP)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    
    
    
    if len(sentences)>top_n:
        for i in range(top_n):
          summarize_text.append(" ".join(ranked_sentence[i][1]))
    else:
        for i in range(round(len(sentences)/2)-1):
          summarize_text.append(" ".join(ranked_sentence[i][1]))

    Resumen=' '.join(summarize_text)

    # Step 5 - Offcourse, output the summarize texr
    #print("Summarize Text: \n", ". ".join(summarize_text))
    return Resumen


df_N2009v3=df_N2009v2.copy()
categoria="articulo"
resumenNR=limpiarNLTK(df_N2009v3[categoria],categoria)
df_N2009v3[categoria]

# let's begin
resumenPred=[]
c=0
for i in df_N2009v3["articulo"]:
    resumenPred.append(generate_summary( i, 4,c,400))
    if c%1000==0:
        print(c)
    c=c+1
    
df_N2009v3["newresumen"]=resumenPred
categoria="newresumen"
resumenNR=limpiarNLTK(df_N2009v3[categoria],categoria)
df_N2009v3["newresumen"]=resumenNR

resumenPred[0]
#listaTxt=resumenPred
listaTxt=df_N2009v3["articulo"]
i=0
def limpiarNLTK(listaTxt,categoria):   
    resumen=[]
    for i in range (45086):
        text_tokens = word_tokenize(listaTxt[i])
        tokens_without_sw = [word for word in text_tokens if not word in stopSpanish]
        tokens_without_sw = [word.lower() for word in tokens_without_sw if word.isalpha()]
        resumen.append(' '.join(tokens_without_sw))
    return resumen
df_N2009v2["articulo"]=resumen
df_N2009v2.to_csv('D:/Computo Estadistico/semestre 3/ciencia de datos 2/newresumenbaseline2.csv', sep=';')


len(df_N2009v3["articulo"][5].split())
len(df_N2009v3["newresumen"][5].split())
