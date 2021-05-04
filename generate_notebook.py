#%%[markdown]
## Carregar base de dados para o SQLite
#Definir localização da base de dados 
#%%
path_to_database='data/raw/elo7_recruitment_dataset.csv'
#%%[markdown]
#Definir localização onde SQLite vai ser guardado, é recomendavel usar uma partição
#mapeada em RAM para aumentar a performance (exemplo /dev/shm)
#%%
path_to_sqlite='data/interim/database.sqlite3' #Store the database in ram partition (as /dev/shm) to increase the performance


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

def create_schema_for_table___word_typed_in_query___query_elo7(path_to_sqlite3):
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur.execute("DROP TABLE IF EXISTS word_typed_in_query___query_elo7;")
    cur.execute("""
    CREATE TABLE word_typed_in_query___query_elo7 (
        word_typed_in_query_id                       INTEGER REFERENCES word_typed_in_query (word_typed_in_query_id) ON DELETE CASCADE
                                                                                                                 ON UPDATE CASCADE,
        querys_elo7_id                               INTEGER REFERENCES query_elo7 (querys_elo7_id) ON DELETE CASCADE
                                                                                                ON UPDATE CASCADE,
        word_typed_in_query___query_elo7_id INTEGER PRIMARY KEY AUTOINCREMENT
    );

    """)
    conn.commit()
    conn.close()


def create_schema_for_table___vector_element(path_to_sqlite3):
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur.execute("DROP TABLE IF EXISTS vector_element;")
    sql="""
    CREATE TABLE vector_element (
    vector_element_id  INTEGER       PRIMARY KEY AUTOINCREMENT,
    querys_elo7_id     INTEGER       REFERENCES query_elo7 (querys_elo7_id) ON DELETE CASCADE
                                                                            ON UPDATE CASCADE,
    position_in_vector INT,
    word               VARCHAR (256),
    value              DOUBLE
    );
    """
    cur.execute(sql)
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

def word_typed_in_query___query_elo7(path_to_sqlite3):
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
            INSERT INTO word_typed_in_query___query_elo7 (
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


create_schema_for_tables_that_associate_words_in_querys_with_querys_typed(path_to_sqlite)
populate_table__word_typed_in_query(path_to_sqlite)
create_schema_for_table___word_typed_in_query___query_elo7(path_to_sqlite)
word_typed_in_query___query_elo7(path_to_sqlite)
create_schema_for_table___vector_element(path_to_sqlite)


#%%[markdown]
## Análise exploratoria
### Análises das palavras dígitadas nas consultas
###### Para estas análises foram retiradas as palavras conhecidas em *PLN* (processamento de linguagem natural) como *stopwords*, *stopwords* são palavras muito comuns que não adicionam significado para o texto (preposições por exemplo)
#Contagem do número de palavras distintas que já foram digitadas por
#usuarios em consultadas, admitindo que nos modelos que vão ser desenvolvidos as
#palavras vão ser as variáveis de entrada, estas são as variáveis que primeiro vão ser análisadas.
#%%
conn=sqlite3.connect(path_to_sqlite)
sql="""SELECT COUNT(DISTINCT word)        
        FROM word_typed_in_query
        WHERE word NOT IN ({stopwords})
    """.format(stopwords=str(list(stopwords.words('portuguese')))[1:-1])
print("Número de palavras distintas já digitadas: "+str(conn.execute(sql).fetchall()[0][0]))
conn.close()

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
    WITH word_typed_in_query___query_elo7_distinct AS (
        SELECT DISTINCT word_typed_in_query_id, querys_elo7_id 
        FROM  word_typed_in_query___query_elo7
    )
    SELECT COUNT(querys_elo7_id) AS numbero_de_consultas_onde_a_palavra_foi_digitada,
           word_typed_in_query.word AS palavra
    FROM word_typed_in_query___query_elo7_distinct
    INNER JOIN word_typed_in_query ON word_typed_in_query.word_typed_in_query_id=word_typed_in_query___query_elo7_distinct.word_typed_in_query_id
    WHERE  word_typed_in_query.word NOT IN ({stopwords})
    GROUP BY word_typed_in_query___query_elo7_distinct.word_typed_in_query_id
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

# %% [markdown]
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
# A seguir um plot que mostra quantas consultas são cobertas por grupos com 
# as N palavras mais frequentes.

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

####df_with_number_of_queries_coverage_by_groups_with_the_N_most_frequent_words=number_of_queries_coverage_by_groups_with_the_N_most_frequent_words(path_to_sqlite,path_to_database)
####sns.lineplot(data=df_with_number_of_queries_coverage_by_groups_with_the_N_most_frequent_words, x="grupo com as N palavras mais frequentes", y="número de consultas cobertas pelo grupo")

# %%
####last_row_for_infomation_about_group_of_words=(df_with_number_of_queries_coverage_by_groups_with_the_N_most_frequent_words.values)[-1]
####print ("Quantidade consultas cobertas pelo grupo com as {num_of_words} palavras mais frequentes: {num_of_querys}".format(num_of_words=last_row_for_infomation_about_group_of_words[0],num_of_querys=last_row_for_infomation_about_group_of_words[1]))



# %%[markdown]
# Observando o gráfico anterior observa-se que as 384 das palavras mais frequentes
# estão em 35591 das 38507 consultas que estão disponiveis na base dados fornecida,
# aproximadamente 92% de cobertura. Deste modo os modelos de IA desenvolvidos caso 
# recebam uma consulta/query como uma de suas entradas vão analisa apenas estas 384 
# palavras.


# %%[markdown]
### Criação de vetor médio de palavras consultadas para cada categoria .
# Para fazer isso devem ser calculados histogramas das palavras que cada produto
# da base de dados possui com base nas consultas associadas ao produto para então, calcular a média
# dos histogramas dos produtos associados a uma categoria.
# Os histogramas podem ser feitos seguindo os seguintes passos: <br>
# <ul>
# <li> 
#   Com base nas 384 palavras mais frequentes associamos cada palavra a uma posição 
#   vetores de 384 elementos onde, cada produto que esta base de dados vai ter um vetor destes. 
#   Estes são os vetores que guardam o histograma dos produtos.
# </li>
# <li>
#    Percorrer todas as consultas que existem para cada um dos produtos cadastrados na
#    base de dados e sempre que achar uma palavra que associada ao vetor de histograma do produto
#    incrementar o valor do elemento do vetor associado a palavra.
# </li>
#</ul>


#%%
def populate_table____vector_element(path_to_sqlite3):
    ranking_for_occurrence_words_in_querys=list((count_number_of_times_that_word_appear_in_query(path_to_sqlite3)['palavra']).values)[:384]
    ranking_for_occurrence_words_in_querys.sort()
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur2=conn.cursor()
    cur.execute("SELECT querys_elo7_id,query FROM query_elo7")
    for line in cur:
        query_words= word_tokenize(line[1])
        elementsToInsert=""
        for i in range(len(ranking_for_occurrence_words_in_querys)):
            number_of_times=query_words.count(ranking_for_occurrence_words_in_querys[i])
            elementsToInsert=elementsToInsert+"({querys_elo7_id},{position_in_vector},'{word}',{number_of_times}),".format(querys_elo7_id=line[0],position_in_vector=i,word=ranking_for_occurrence_words_in_querys[i],number_of_times=number_of_times)
        
        cur2.execute("INSERT INTO vector_element (querys_elo7_id,position_in_vector,word,value) VALUES "+elementsToInsert[:-1])
        conn.commit()
    conn.close()

populate_table____vector_element(path_to_sqlite)
"""
WITH TEMP3 AS (
SELECT query_elo7.product_id AS product_id,    
       position_in_vector AS position_in_vector,
       word AS word,
       category,
       SUM(value) AS CC
  FROM vector_element
  INNER JOIN query_elo7 ON query_elo7.querys_elo7_id=vector_element.querys_elo7_id

  GROUP BY query_elo7.product_id,position_in_vector,word,category
)


SELECT category,AVG(CC)
FROM TEMP3
GROUP BY category
ORDER BY category,position_in_vector
"""
exit(0)





# %%[markdown]
### Análises dos atributos dos produtos existentes (peso, preço e entrega rápida).



#### A seguir histogramas para diferentes faixas  peso que os produtos possuem em cada categoria.
# %%[markdown]
# Histogramas com a distribuição de peso por categoria.
#%%
conn=sqlite3.connect(path_to_sqlite)

df=pd.read_sql_query("""SELECT DISTINCT product_id, category, weight FROM query_elo7 WHERE weight""",conn)
sns.histplot(hue="category", x="weight", data=df,bins=10)
conn.close()
# %%
# %%[markdown]
# Histogramas com a distribuição de peso por categoria com limite de peso de até 40, isso é equivalente a um zoom no inicio no eixo X dos histogramas do gráfico anterior.
#%%
conn=sqlite3.connect(path_to_sqlite)

df=pd.read_sql_query("""SELECT DISTINCT product_id, category, weight FROM query_elo7 WHERE weight<40""",conn)
#ax = sns.boxplot(x="category", y="weight", data=df)
sns.histplot(hue="category", x="weight", data=df,bins=10)
conn.close()


# %%[markdown]
#### A seguir histogramas para diferentes faixas preço que os produtos possuem em cada categoria.
# %%[markdown]
# Histogramas com a distribuição de preço por categoria.
#%%
conn=sqlite3.connect(path_to_sqlite)

df=pd.read_sql_query("""SELECT DISTINCT product_id, category, price FROM query_elo7 WHERE price""",conn)
sns.histplot(hue="category", x="price", data=df,bins=10)
conn.close()

# %%[markdown]
# Histogramas com a distribuição de peso por categoria com limite de preço de até 100, isso é equivalente a um zoom no inicio do eixo X dos histogramas do gráfico anterior.
#%%
conn=sqlite3.connect(path_to_sqlite)

df=pd.read_sql_query("""SELECT DISTINCT product_id, category, price FROM query_elo7 WHERE price<200  """,conn)
#ax = sns.boxplot(x="category", y="weight", data=df)
sns.histplot(hue="category", x="price", data=df,bins=10)
conn.close()


# %%[markdown]
# Nos histogramas do gráfico anterior a categoria "Lembrancinhas" esta atrapalhando 
# a visualizaçao das distribuição de preços das outras categorias, ela vai ser retirada
# do próximo gráfico então. O gráifco a seguir é um replot dos histogramas do gráfico 
# antorior porem sem a categoria "Lembrancinhas".

#%%
conn=sqlite3.connect(path_to_sqlite)

df=pd.read_sql_query("""SELECT DISTINCT product_id, category, price FROM query_elo7 WHERE  price<200 AND category!='Lembrancinhas'  """,conn)
#ax = sns.boxplot(x="category", y="weight", data=df)
sns.histplot(hue="category", x="price", data=df,bins=10)
conn.close()


# %%[markdown]
#### Análise sobre a distribuição de peso e preço 
# Como pode ser observado nem o preço nem o peso  dos produtos segue uma distribuição 
# normal portanto, a melhor estrategia para discretizar estes valores seria clusteriza-los,
# via algum algoritmo de aprendizado de aprendizado não supervisionado como o K-means.
# A discretização de valores continuos é uma estrategia que pode melhorar qualidade dos
# modelos de IA desenvolvidos.

# %%[markdown]
#### Análise sobre a distribuição do atributo envio expresso

#%%

conn=sqlite3.connect(path_to_sqlite)
df=pd.read_sql_query("""SELECT DISTINCT product_id, category, express_delivery FROM query_elo7 WHERE  price<200 AND category!='Lembrancinhas'  """,conn)


df['express_delivery']=df['express_delivery'].apply(lambda x: 'yes' if x>0.0 else 'no')
categories=df['category'].unique()
for category in categories:
    print("Distribuição para a categoria:"+str(category)+"\n")
    dfT=df[df['category']==category]
    sns.histplot( x="express_delivery", data=dfT.sort_values(by=['express_delivery']), stat='probability',discrete=True, shrink=.8)
    plt.show()
    print("\n\n\n\n")


conn.close()
# %%[markdown]
# Como pode ser observado existem diferentes distribuições de envio expresso
# por categoria, isso faz este atributo interessante para usar em um 
# classificador de categorias.


# %%[markdown]
