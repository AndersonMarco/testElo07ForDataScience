

#%%[markdown]
## Carregar base de dados para o SQLite**
#Definir localização da base de dados 
#%%
path_to_database='data/raw/elo7_recruitment_dataset.csv'
#%%[markdown]
#Definir localização onde SQLite vai ser guardado
#%%
path_to_sqlite='data/interim/database.sqlite3'


#%%[markdown]
#Passar os dados de CSV para SQLITE, isso é para facilitar algumas
#análises porque o SQLITE é uma linguagem de consulta de dados que permite
#facíl manipulação dos dados.
#%%
import src.data.load_data_to_sqlite_dataset as load_dataset
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
#%%
def pass_csv_database_to_sqlite3(path_to_sqlite3,path_raw_data_in_csv):

    conn = sqlite3.connect(path_to_sqlite3)
    df=pd.read_csv(path_raw_data_in_csv)
    df.to_sql("query_elo7", conn, if_exists="replace",index_label='querys_elo7_id')
    conn.close()

pass_csv_database_to_sqlite3(path_to_sqlite,path_to_database)

#####load_dataset.create_sqlite_schema()

#%%[markdown]
#Associar individualmente cada palavras digitadas nas consultas com as 
#consultas em que foram digitadas, isso é util para futuras análises
#estatísticas
#%%

def create_schema_for_tables_that_associate_words_in_querys_with_querys_typed(path_to_sqlite3):
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur.execute("DROP TABLE IF EXISTS word_typed_in_query;")
    cur.execute("""
    CREATE TABLE word_typed_in_query (
        word                   VARCHAR (256),
        word_typed_in_query_id INTEGER       PRIMARY KEY AUTOINCREMENT
    );
    """)
    conn.commit()
    conn.close()

def create_schema_for_table___word_typed_in_query___word_typed_in_query(path_to_sqlite3):
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur.execute("DROP TABLE IF EXISTS word_typed_in_query___word_typed_in_query;")
    cur.execute("""
    CREATE TABLE word_typed_in_query___word_typed_in_query (
        word_typed_in_query_id                       INTEGER REFERENCES word_typed_in_query (word_typed_in_query_id) ON DELETE CASCADE
                                                                                                                 ON UPDATE CASCADE,
        querys_elo7_id                               INTEGER REFERENCES query_elo7 (querys_elo7_id) ON DELETE CASCADE
                                                                                                ON UPDATE CASCADE,
        word_typed_in_query___word_typed_in_query_id INTEGER PRIMARY KEY AUTOINCREMENT
    );

    """)
    conn.commit()
    conn.close()

def populate_table__word_typed_in_query(path_to_sqlite3):
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur.execute(""" SELECT query
                    FROM query_elo7;
                """)
    words={}
    for line in cur:
        words_line= word_tokenize(line[0])
        for word_line in words_line:
            if(not(word_line in words)):
                words[word_line]=None
    
    for word in words.keys():
        cur.execute("""
                        INSERT INTO word_typed_in_query (
                                                            word
                                  
                                                        )
                                                VALUES (
                                                            '{word}'          
                                                        );
                    """.format(word=word))
    conn.commit()
    conn.close()

def word_typed_in_query___word_typed_in_query(path_to_sqlite3):
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    converter_word_to_word_id_in_table={}
    cur.execute(''' 
        SELECT  word,
                word_typed_in_query_id
        FROM word_typed_in_query;
    ''')
    for line in cur:
        converter_word_to_word_id_in_table[line[0]]=line[1]

    cur.execute(""" SELECT querys_elo7_id,
                           query
                    FROM query_elo7;
                """)
    cur2=conn.cursor()
    for line in cur:
        query_words= word_tokenize(line[1])
        for word in query_words:
            word_id=converter_word_to_word_id_in_table[word]
            sql='''
            INSERT INTO word_typed_in_query___word_typed_in_query (
                                                          word_typed_in_query_id,
                                                          querys_elo7_id
                                                      )
                                                      VALUES (
                                                            {word_typed_in_query_id},
                                                            {querys_elo7_id}
                                                      );
            '''.format(word_typed_in_query_id=word_id, querys_elo7_id=line[0])
            cur2.execute(sql)
    conn.commit()
    conn.close()


#####create_schema_for_tables_that_associate_words_in_querys_with_querys_typed(path_to_sqlite)
#####populate_table__word_typed_in_query(path_to_sqlite)
#####create_schema_for_table___word_typed_in_query___word_typed_in_query(path_to_sqlite)
#####word_typed_in_query___word_typed_in_query(path_to_sqlite)

#%%[markdown]
## Análise exploratoria
### Análises das palavras dígitadas nas consultas
###### Para estas análises foram retiradas as palavras conhecidas em *PLN* (processamento de linguagem natural) como *stopwords*, *stopwords* são palavras muito comuns que não adicionam significado para o texto (preposições por exemplo)
#Contagem do número de palavras distintas que já foram digitadas por
#usuarios em consultadas, admitindo que nos modelos que vão ser desenvolvidos as palavras vão ser as variáveis de entrada, estas são as variáveis que primeiro vão ser análisadas.
#%%
conn=sqlite3.connect(path_to_sqlite)
sql="""SELECT COUNT(DISTINCT word)        
        FROM word_typed_in_query
        WHERE word NOT IN ({stopwords})
    """.format(stopwords=str(list(stopwords.words('portuguese')))[1:-1])
print("Número de palavras distintas já digitadas: "+str(conn.execute(sql).fetchall()[0][0]))


#%%[markdown]
# Por causa da quantidade de palavras que existem é preciso realizar
# uma contagem de frequência de palavras nas consultas, isso pode ser 
# utíl para diminuir quantidade de palavras que são necessárias serem 
# tratadas pelos modelos desenvolvidos. Com menos palavras para se
# preocupar em analisar os modelos podem ser mais acertivos, essa
# é uma tática comum para melhor a qualidade de modelos de IA.

#%%
def count_number_of_times_that_word_appear_in_query(path_to_sqlite3):
    

    sql="""
    WITH word_typed_in_query___word_typed_in_query_distinct AS (
        SELECT DISTINCT word_typed_in_query_id, querys_elo7_id 
        FROM  word_typed_in_query___word_typed_in_query
    )
    SELECT COUNT(querys_elo7_id) AS numbero_de_consultas_onde_a_palavra_foi_digitada,
           word_typed_in_query.word AS palavra
    FROM word_typed_in_query___word_typed_in_query_distinct
    INNER JOIN word_typed_in_query ON word_typed_in_query.word_typed_in_query_id=word_typed_in_query___word_typed_in_query_distinct.word_typed_in_query_id
    WHERE  word_typed_in_query.word NOT IN ({stopwords})
    GROUP BY word_typed_in_query___word_typed_in_query_distinct.word_typed_in_query_id
    ORDER BY COUNT(querys_elo7_id) DESC
    """.format(stopwords=str(list(stopwords.words('portuguese')))[1:-1])
    
    conn = sqlite3.connect(path_to_sqlite3)
    df=pd.read_sql_query(sql,conn)

    conn.close()
    return df
print("Análise de frequência das vinte palavras mais digitadas nas consultas:")
df_number_of_times_for_words_in_querys=count_number_of_times_that_word_appear_in_query(path_to_sqlite)
df_number_of_times_for_words_in_querys.head(20)
# %%[markdown]
# Pode-se notar um decaimento exponencial (muito rápido) na 
# frequencia da palavra mais digitada para vigesima mais digitada. <br>
# Uma melhor análise seria plotar um gráfico para confirmar
# este decaimento exponencial onde, o eixo X representa o
# *ranking* da palavra entre as que mais aparecem e o eixo Y o número de 
# vezes que ela aparece.

# %%
df_number_of_times_for_words_in_querys=count_number_of_times_that_word_appear_in_query(path_to_sqlite)
df_number_of_times_for_words_in_querys=df_number_of_times_for_words_in_querys.reset_index()
df_number_of_times_for_words_in_querys.rename(columns = {'index':'ranking da palavra', 'numbero_de_consultas_onde_a_palavra_foi_digitada':'número de vezes que aparece'}, inplace = True)
sns.lineplot(data=df_number_of_times_for_words_in_querys.reset_index(), x="ranking da palavra", y="número de vezes que aparece")

# %%
# Com as análises apresentadas até agora pode-se dizer que com poucas
# palavras, algumas centenas, esta dentro da maioria das consultas 
# (querys). Para confirmar isso
# é necessario criar grupos de palavras onde o primeiro grupo possui
# apenas a palavras mais frequente, o segundo grupo as duas palavras
# mais frequentes, o terceiro as três palavras mais frequentes e assim 
# por diante, para então analisar em quantas consultas existe pelo uma das 
# palavras de um determinado grupo. Para simplificar a escrita vai ser 
# utilizado o termo *cobertura* neste relatorio tem a seguinte
# definição: "Um grupo de palavras cobre um consulta se e somente se existe 
# pelo menos uma palavra do grupo que é esta dentro da consulta".<br>
# A seguir um plot que mostra quantas consultas são cobertas por grupos <br>
# com as N palavras mais frequentes.

#%%
def words_are_in_query(listOfWordsToSearch, query_string):
    for word in listOfWordsToSearch:
        if(word in query_string):
            return 1
    return 0

def number_of_queries_coverage_by_groups_with_the_N_most_frequent_words(path_to_sqlite3,path_to_csv):
    ranking_for_occurrence_words_in_querys=(count_number_of_times_that_word_appear_in_query(path_to_sqlite3)['palavra']).values
    prototype_for_dataframe_with_result={'grupo com as N palavras mais frequentes':[],'número de consultas cobertas pelo grupo':[]}
    for i in range(384):
        wordGroup=ranking_for_occurrence_words_in_querys[:(i+1)]
        queryBelongToWordGroup=lambda query:  words_are_in_query(wordGroup,query)
        df=pd.read_csv(path_to_csv)
        numberOfQuerysCoverage=np.sum(df['query'].apply(queryBelongToWordGroup).values)
        prototype_for_dataframe_with_result['grupo com as N palavras mais frequentes'].append(i+1)
        prototype_for_dataframe_with_result['número de consultas cobertas pelo grupo'].append(numberOfQuerysCoverage)

    return pd.DataFrame.from_dict(prototype_for_dataframe_with_result)

df_with_number_of_queries_coverage_by_groups_with_the_N_most_frequent_words=number_of_queries_coverage_by_groups_with_the_N_most_frequent_words(path_to_sqlite,path_to_database)
sns.lineplot(data=df_with_number_of_queries_coverage_by_groups_with_the_N_most_frequent_words, x="grupo com as N palavras mais frequentes", y="número de consultas cobertas pelo grupo")

# %%
last_row_for_infomation_about_group_of_words=(df_with_number_of_queries_coverage_by_groups_with_the_N_most_frequent_words.values)[-1]
print ("Quantidade consultas cobertas pelo grupo com as {num_of_words} palavras mais frequentes: {num_of_querys}".format(num_of_words=last_row_for_infomation_about_group_of_words[0],num_of_querys=last_row_for_infomation_about_group_of_words[1]))



# %%
# Observando o gráfico anterior observa-se que as 384 das palavras mais frequentes
# estão em 35591 das 38507 consultas que estão disponiveis na base dados fornecida,
# aproximadamente 92% de cobertura. Deste modo os modelos de IA desenvolvidos caso 
# recebam uma consulta/query como uma de suas entradas vão analisa apenas estas 384 palavras.

