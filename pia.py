# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:09:28 2022

@author: david
"""
      
####################
import sqlite3
dbfile="D:/Computo Estadistico/PIA-CD2/Noticias Reforma DB.db"
conn = sqlite3.connect(dbfile)
c=conn.cursor()
table_list = [a for a in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]

#cargando data
c.execute(" SELECT * FROM Noticias1993")
D1993=c.fetchall()
c.execute(" SELECT * FROM Noticias1994")
D1994=c.fetchall()
c.execute(" SELECT * FROM Noticias1995")
D1995=c.fetchall()
conn.close()

dbfile="D:/Computo Estadistico/PIA-CD2/Noticias Reforma DB.db"
conn = sqlite3.connect(dbfile)
c=conn.cursor()
table_list = [a for a in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]

#cargando data
c.execute(" SELECT * FROM Noticias1993")
D1993=c.fetchall()
c.execute(" SELECT * FROM Noticias1994")
D1994=c.fetchall()
c.execute(" SELECT * FROM Noticias1995")
D1995=c.fetchall()
conn.close()

###########

import pandas as pd
import sqlite3
import sqlalchemy 

dbfile1 = 'D:/Computo Estadistico/PIA-CD2/NoticiasReforma2009.db'
dbfile2 = 'D:/Computo Estadistico/PIA-CD2/NoticiasReforma2001.db' ## Reforma2001, Noticias2001
dbfile3="D:/Computo Estadistico/PIA-CD2/Noticias Reforma DB.db" #Noticias1993',), ('Noticias1994',), ('Noticias1995'


try:
    con = sqlite3.connect(dbfile1)    
except Exception as e:
    print(e)
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#print(f"Table Name : {cursor.fetchall()}")
df_N2009 = pd.read_sql_query('SELECT * FROM Noticias2009', con)
df_R2009 = pd.read_sql_query('SELECT * FROM Reforma2009', con)
con.close()


try:
    con = sqlite3.connect(dbfile2)    
except Exception as e:
    print(e)
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#print(f"Table Name : {cursor.fetchall()}")
df_N2001 = pd.read_sql_query('SELECT * FROM Noticias2001', con)
df_R2001 = pd.read_sql_query('SELECT * FROM Reforma2001', con)
con.close()




try:
    con = sqlite3.connect(dbfile2)    
except Exception as e:
    print(e)
cursor = con.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#print(f"Table Name : {cursor.fetchall()}")
df_N1993 = pd.read_sql_query('SELECT * FROM Noticias1993', con)
df_N1994 = pd.read_sql_query('SELECT * FROM Noticias1994', con)
df_N1995 = pd.read_sql_query('SELECT * FROM Noticias1995', con)
con.close()


#Codigo
df_N2009.shape

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import nltk
#Importamos las funciones vistas en clase
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')

stopSpanish=set(stopwords.words('spanish'))

resumen=[]
for i in range (1):
  text_tokens = word_tokenize(df_N2009['articulo'][i])
  tokens_without_sw = [word for word in text_tokens if not word in stopSpanish]
  resumen.append((tokens_without_sw))
  
resumen=[]
for i in range (61035):
  text_tokens = word_tokenize(df_N2009['articulo'][i])
  tokens_without_sw = [word for word in text_tokens if not word in stopSpanish]
  tokens_without_sw = [word.lower() for word in tokens_without_sw if word.isalpha()]
  resumen.append(' '.join(tokens_without_sw))
  
!pip install pysummarization

from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor

# Object of automatic summarization.
auto_abstractor = AutoAbstractor()
# Set tokenizer.
auto_abstractor.tokenizable_doc = SimpleTokenizer()
# Set delimiter for making a list of sentence.
auto_abstractor.delimiter_list = [".", "\n"]
# Object of abstracting and filtering document.
abstractable_doc = TopNRankAbstractor()
# Summarize document
summ=[]
for i in range(61035):
  document=resumen[i]
  result_dict = auto_abstractor.summarize(document, abstractable_doc)
  summ.append(result_dict['summarize_result'])