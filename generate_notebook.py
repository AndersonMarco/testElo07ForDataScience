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
import sklearn 
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('stopwords')
#%%
def xorshift32(x):    
    for i in range(16):
        x ^= (x << 13) &4294967295
        x ^= (x >> 17) &4294967295
        x ^= (x << 5)&4294967295
    return  x%20

def pass_csv_database_to_sqlite3(path_to_sqlite3,path_raw_data_in_csv):
    conn = sqlite3.connect(path_to_sqlite3)
    df=pd.read_csv(path_raw_data_in_csv)
    
    df=df.reset_index()
    df=df.rename(columns = {"index": "query_elo7_id"})
    df['hash_for_query_elo7_id']=df.apply((lambda row: xorshift32(row.query_elo7_id)), axis = 1)
    df['hash_for_product_id']=df.apply((lambda row: xorshift32(row.product_id)), axis = 1)
    df.to_sql("query_elo7", conn, if_exists="replace",index=False)
    conn.commit()
    cur=conn.cursor()
    cur.execute("""
    CREATE INDEX index_query_elo7_querys_elo7_id ON query_elo7 (
        query_elo7_id
    );

    """)
    cur.execute("""
    CREATE INDEX index_query_elo7_hash_for_query_elo7_id ON query_elo7 (
        hash_for_query_elo7_id
    );

    """)
    conn.commit()
    conn.close()

######pass_csv_database_to_sqlite3(path_to_sqlite,path_to_database)

######load_dataset.create_sqlite_schema()

#%%[markdown]
#Associar individualmente cada palavras digitadas nas consultas com as 
#consultas em que foram digitadas, isso é util para futuras análises
#estatísticas
#%%

def create_schema_for_tables_that_associate_words_in_querys_with_querys_typed(path_to_sqlite3):
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur.execute("DROP TABLE IF EXISTS word_typed_in_query;")
    cur.execute("VACUUM;")
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
    cur.execute("VACUUM;")
    cur.execute("""
    CREATE TABLE word_typed_in_query___query_elo7 (
        word_typed_in_query_id                       INTEGER REFERENCES word_typed_in_query (word_typed_in_query_id) ON DELETE CASCADE
                                                                                                                 ON UPDATE CASCADE,
        query_elo7_id                               INTEGER REFERENCES query_elo7 (query_elo7_id) ON DELETE CASCADE
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
    cur.execute("VACUUM;")
    sql="""
    CREATE TABLE vector_element (
    vector_element_id  INTEGER       PRIMARY KEY AUTOINCREMENT,
    query_elo7_id     INTEGER       REFERENCES query_elo7 (query_elo7_id) ON DELETE CASCADE
                                                                            ON UPDATE CASCADE,
    position_in_vector INT,
    word               VARCHAR (256),
    value              DOUBLE,
    hash_for_query_elo7_id INTEGER
    );
    """
    cur.execute(sql)
    cur.execute("""
    CREATE INDEX index_vector_element___query_elo7_id ON vector_element (
        query_elo7_id
    );

    """)
    cur.execute("""
    CREATE INDEX index_vector_element___hash_for_query_elo7_id ON vector_element (
        hash_for_query_elo7_id
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

    cur.execute(""" SELECT query_elo7_id,
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
                                                          query_elo7_id
                                                      )
                                                      VALUES (
                                                            {word_typed_in_query_id},
                                                            {query_elo7_id}
                                                      );
            '''.format(word_typed_in_query_id=word_id, query_elo7_id=line[0])
            cur2.execute(sql)
    conn.commit()
    conn.close()


######create_schema_for_tables_that_associate_words_in_querys_with_querys_typed(path_to_sqlite)
######populate_table__word_typed_in_query(path_to_sqlite)
######create_schema_for_table___word_typed_in_query___query_elo7(path_to_sqlite)
######word_typed_in_query___query_elo7(path_to_sqlite)
######create_schema_for_table___vector_element(path_to_sqlite)


#%%[markdown]
## Análise exploratoria
### Análises das palavras dígitadas nas consultas
###### Para estas análises foram retiradas as palavras conhecidas em *PLN* (processamento de linguagem natural) como *stopwords*, *stopwords* são palavras muito comuns que não adicionam significado para o texto (preposições por exemplo)
#Contagem do número de palavras distintas que já foram digitadas por
#usuarios em consultadas, admitindo que nos modelos que vão ser desenvolvidos as
#palavras vão ser as variáveis de entrada, estas são as variáveis que primeiro vão ser análisadas.
#%%
######conn=sqlite3.connect(path_to_sqlite)
######sql="""SELECT COUNT(DISTINCT word)        
######        FROM word_typed_in_query
######        WHERE word NOT IN ({stopwords})
######    """.format(stopwords=str(list(stopwords.words('portuguese')))[1:-1])
######print("Número de palavras distintas já digitadas: "+str(conn.execute(sql).fetchall()[0][0]))
######conn.close()

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
        SELECT DISTINCT word_typed_in_query_id, query_elo7_id 
        FROM  word_typed_in_query___query_elo7
    )
    SELECT COUNT(query_elo7_id) AS numbero_de_consultas_onde_a_palavra_foi_digitada,
           word_typed_in_query.word AS palavra
    FROM word_typed_in_query___query_elo7_distinct
    INNER JOIN word_typed_in_query ON word_typed_in_query.word_typed_in_query_id=word_typed_in_query___query_elo7_distinct.word_typed_in_query_id
    WHERE  word_typed_in_query.word NOT IN ({stopwords})
    GROUP BY word_typed_in_query___query_elo7_distinct.word_typed_in_query_id
    ORDER BY COUNT(query_elo7_id) DESC
    """.format(stopwords=str(list(stopwords.words('portuguese')))[1:-1])
    
    conn = sqlite3.connect(path_to_sqlite3)
    df=pd.read_sql_query(sql,conn)

    conn.close()
    return df
#####print("Análise de frequência das vinte palavras mais digitadas nas consultas:")
#####df_number_of_times_for_words_in_querys=count_number_of_times_that_word_appear_in_query(path_to_sqlite)
#####df_number_of_times_for_words_in_querys.head(20)
# %%[markdown]
# Pode-se notar um decaimento exponencial (muito rápido) na 
# frequencia da palavra mais digitada para vigesima mais digitada. <br>
# Uma melhor análise seria plotar um gráfico para confirmar
# este decaimento exponencial onde, o eixo X representa o
# *ranking* da palavra entre as que mais aparecem e o eixo Y o número de 
# vezes que ela aparece.

# %%
######df_number_of_times_for_words_in_querys=count_number_of_times_that_word_appear_in_query(path_to_sqlite)
#####df_number_of_times_for_words_in_querys=df_number_of_times_for_words_in_querys.reset_index()
#####df_number_of_times_for_words_in_querys.rename(columns = {'index':'ranking da palavra', 'numbero_de_consultas_onde_a_palavra_foi_digitada':'número de vezes que aparece'}, inplace = True)
#####sns.lineplot(data=df_number_of_times_for_words_in_querys.reset_index(), x="ranking da palavra", y="número de vezes que aparece")

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
### Análises dos atributos dos produtos existentes (peso, preço e entrega rápida).



#### A seguir histogramas para diferentes faixas  peso que os produtos possuem em cada categoria.
# %%[markdown]
# Histogramas com a distribuição de peso por categoria.
#%%
conn=sqlite3.connect(path_to_sqlite)

#####df=pd.read_sql_query("""SELECT DISTINCT product_id, category, weight FROM query_elo7 WHERE weight""",conn)
####sns.histplot(hue="category", x="weight", data=df,bins=10)
conn.close()
# %%
# %%[markdown]
# Histogramas com a distribuição de peso por categoria com limite de peso de até 40, isso é equivalente a um zoom no inicio no eixo X dos histogramas do gráfico anterior.
#%%
conn=sqlite3.connect(path_to_sqlite)

#######df=pd.read_sql_query("""SELECT DISTINCT product_id, category, weight FROM query_elo7 WHERE weight<40""",conn)

####sns.histplot(hue="category", x="weight", data=df,bins=10)
conn.close()


# %%[markdown]
#### A seguir histogramas para diferentes faixas preço que os produtos possuem em cada categoria.
# %%[markdown]
# Histogramas com a distribuição de preço por categoria.
#%%
conn=sqlite3.connect(path_to_sqlite)

####df=pd.read_sql_query("""SELECT DISTINCT product_id, category, price FROM query_elo7 WHERE price""",conn)
####sns.histplot(hue="category", x="price", data=df,bins=10)
conn.close()

# %%[markdown]
# Histogramas com a distribuição de peso por categoria com limite de preço de até 100, isso é equivalente a um zoom no inicio do eixo X dos histogramas do gráfico anterior.
#%%
conn=sqlite3.connect(path_to_sqlite)

######df=pd.read_sql_query("""SELECT DISTINCT product_id, category, price FROM query_elo7 WHERE price<200  """,conn)
#ax = sns.boxplot(x="category", y="weight", data=df)
#####sns.histplot(hue="category", x="price", data=df,bins=10)
conn.close()


# %%[markdown]
# Nos histogramas do gráfico anterior a categoria "Lembrancinhas" esta atrapalhando 
# a visualizaçao das distribuição de preços das outras categorias, ela vai ser retirada
# do próximo gráfico então. O gráifco a seguir é um replot dos histogramas do gráfico 
# antorior porem sem a categoria "Lembrancinhas".

#%%
conn=sqlite3.connect(path_to_sqlite)

####df=pd.read_sql_query("""SELECT DISTINCT product_id, category, price FROM query_elo7 WHERE  price<200 AND category!='Lembrancinhas'  """,conn)
#ax = sns.boxplot(x="category", y="weight", data=df)
####sns.histplot(hue="category", x="price", data=df,bins=10)
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

######conn=sqlite3.connect(path_to_sqlite)
######df=pd.read_sql_query("""SELECT DISTINCT product_id, category, express_delivery FROM query_elo7 WHERE  price<200 AND category!='Lembrancinhas'  """,conn)


######df['express_delivery']=df['express_delivery'].apply(lambda x: 'yes' if x>0.0 else 'no')
######categories=df['category'].unique()
######for category in categories:
    ######print("Distribuição para a categoria:"+str(category)+"\n")
    ######dfT=df[df['category']==category]
    ######sns.histplot( x="express_delivery", data=dfT.sort_values(by=['express_delivery']), stat='probability',discrete=True, shrink=.8)
    ######plt.show()
    ######print("\n\n\n\n")


conn.close()
# %%[markdown]
# Como pode ser observado existem diferentes distribuições de envio expresso
# por categoria, isso faz este atributo interessante para usar em um 
# classificador de categorias.


#%%[markdown]
## Sistema de Classificação de Produtos
#### Histograma de palavras dos produtos associados as palavras digitadas as buscas
# Esse histograma foi uma das metricas criadas que servem de entrada para os modelos descritos
# nas seções  **Encontrar o melhor modelo de predição de categorias com base na metodologia K-Fold**, **Sistema de termos de busca** e **Colaboração entre os sistemas**.
# Para fazer essa métrica devem ser calculados os histogramas das palavras que cada produto
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
# A seguir o código que cria os histogramas para cada consulta.

#%%
def populate_table____vector_element(path_to_sqlite3):
    ranking_for_occurrence_words_in_querys=list((count_number_of_times_that_word_appear_in_query(path_to_sqlite3)['palavra']).values)[:384]
    ranking_for_occurrence_words_in_querys.sort()
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur2=conn.cursor()
    cur.execute("SELECT query_elo7_id,query FROM query_elo7")
    for line in cur:
        query_words= word_tokenize(line[1])
        elementsToInsert=""
        for i in range(len(ranking_for_occurrence_words_in_querys)):
            number_of_times=query_words.count(ranking_for_occurrence_words_in_querys[i])
            elementsToInsert=elementsToInsert+"({query_elo7_id},{position_in_vector},'{word}',{number_of_times},{hash_for_query_elo7_id}),".format(query_elo7_id=line[0],position_in_vector=i,word=ranking_for_occurrence_words_in_querys[i],number_of_times=number_of_times,hash_for_query_elo7_id=xorshift32(line[0]))
        
        cur2.execute("INSERT INTO vector_element (query_elo7_id,position_in_vector,word,value,hash_for_query_elo7_id) VALUES "+elementsToInsert[:-1])
        conn.commit()
    conn.close()

######populate_table____vector_element(path_to_sqlite)



######df_histogram_categories=calculate_histogram_of_words_for_categories(path_to_sqlite)

#%%[markdown]
#### Variáveis utilizadas para a criação do sistema
# Para a criação de sistema foram utilizados os campos price, weight e express_delivery das consultas  
# e o histograma de palavras dos produtos associados as palavras digitadas as buscas.
#### Encontrar o melhor modelo de predição de categorias com base na metodologia K-Fold 
# A maioria dos passos a seguir vão ser repetidos varias vezes com o objetivo de encontrar o melho modelo, esta 
# repetição usa a metodoliga K-Fold e ela consiste de:
# <ul>
# <li>
#    Dividir um conjunto de dados em validação e não validação de forma aleatória.
# </li>
# <li> 
#   Dividir a não validação do conjunto de dados em treinamento e teste, com 86.67% do conjunto como treinamento e 13.33% teste.
# </li>
# <li>
#    Treinar um modelo no conjunto de treinamento e testa-lo no conjunto de teste e no conjunto de validação.
# </li>
# <li>
#    Repartir a não validação entre treinamento e teste de modo que 13.33% do conjunto de treino anterior vire o conjunto de  teste atual e os elementos do conjunto de teste anterior entrem no conjunto de treinamento atual.
# </li>
# <li>
#    Voltar para o passo 3 até todos os elementos do conjunto de não validação tenham sido utilizados para treinar ao menos um modelo.
# </li>
# <li>
#    Utilizar o modelo que obteve melhor resultado no conjunto de validação.
# </li>
# </ul>
#
# Para facilitar o escolha dos conjuntos que dados que vão ser parte do treino, teste e 
# validação as tabelas vector_element e query_elo7 possuem uma coluna chamada hash_for_query_elo7_id
# onde o valor de hash vai de 0 a 19 deste modo, a divisão pode ser feita com base nos valores do
#  hash que devem ser utilizados para que elementos destas tabelas façam parte dos conjuntos de treino,
# teste e validação. Para a seleção do melhor modelo a métrica utilizada foi a media das acuracias 
# alcançadas para classificar cada categoria.

# A seguir a matriz de confusão e a acurácia média para o melhor de modelo de predição encontrado para a elaboração do sistema:
#%%

def transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products):
    dataFrameInDict={'price':[],'weight':[],'express_delivery':[]}
    y=[]
    for i in range(384):
        dataFrameInDict['vector_e'+str(i)]=[]

    for product_id in products:
        product=products[product_id]
        dataFrameInDict['price'].append(product['price'])
        dataFrameInDict['weight'].append(product['weight'])
        dataFrameInDict['express_delivery'].append(product['express_delivery'])
        for i in range(384):
            dataFrameInDict['vector_e'+str(i)].append(product['vector'][i])

        y.append(product['category'])

    df=pd.DataFrame.from_dict(dataFrameInDict)
    df['weight'].fillna(value=df['weight'].mean(), inplace=True)
    df['price'].fillna(value=df['price'].mean(), inplace=True)
    df['price'].fillna(value=0, inplace=True)
    return {'dataframeX':df,'Y':y}

def pass_sql_query_to_get_data_to_train_and_test_and_validation_to_dictonaries_where_the_key_is_the_product_id(path_to_sqlite3,query):
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    
    
    information_about_products={}
    sql="""
    SELECT DISTINCT product_id,       
           category,
           weight,
           price,
           express_delivery
    FROM query_elo7;
    """
    cur.execute(sql)
    for row in cur:
        information_about_products[row[0]]={"category":row[1],"weight":row[2],"price":row[3],'express_delivery':row[4]}

    cur.execute(query)
    productsDict={}

    for row in cur:
        if(not (row[0]  in productsDict)):
            productsDict[row[0]]={'vector':np.zeros(384),
                                   'category':information_about_products[row[0]]['category'],
                                   'weight':information_about_products[row[0]]['weight'],
                                   'price':information_about_products[row[0]]['price'],
                                   'express_delivery':information_about_products[row[0]]['express_delivery']
                                   }

        productsDict[row[0]]['vector'][row[1]]=row[2]
    conn.close()
    return productsDict

def get_data_to_validation(path_to_sqlite3):
    sql="""
    WITH vector_products AS (
        SELECT query_elo7.product_id,
            vector_element.position_in_vector,       
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id
        WHERE query_elo7.hash_for_product_id>=15

  
        GROUP BY query_elo7.product_id,vector_element.position_in_vector
        ORDER BY query_elo7.product_id,vector_element.position_in_vector
    )  
  
    SELECT product_id, 
           position_in_vector,
           value
    FROM vector_products
    WHERE value>0
    """
    
    products_dict_validation=pass_sql_query_to_get_data_to_train_and_test_and_validation_to_dictonaries_where_the_key_is_the_product_id(path_to_sqlite3,sql)
    return transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_validation)

def get_data_to_train_and_test(path_to_sqlite3,folder):
    sql="""
    WITH vector_products AS (
        SELECT query_elo7.product_id,
            vector_element.position_in_vector,       
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id
        WHERE query_elo7.hash_for_product_id<15 AND query_elo7.hash_for_product_id  NOT IN ({folder1},{folder2})
  
        GROUP BY query_elo7.product_id,vector_element.position_in_vector
        ORDER BY query_elo7.product_id,vector_element.position_in_vector
    )  
  
    SELECT product_id, 
           position_in_vector,
           value
    FROM vector_products
    WHERE value>0
    """.format(folder1=folder*2,folder2=(folder*2)+1)
    
    products_dict_train=pass_sql_query_to_get_data_to_train_and_test_and_validation_to_dictonaries_where_the_key_is_the_product_id(path_to_sqlite3,sql)



    sql="""
    WITH vector_products AS (
        SELECT query_elo7.product_id,
            vector_element.position_in_vector,       
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id
        WHERE query_elo7.hash_for_product_id<15 AND query_elo7.hash_for_product_id  IN ({folder1},{folder2})
  
        GROUP BY query_elo7.product_id,vector_element.position_in_vector
        ORDER BY query_elo7.product_id,vector_element.position_in_vector
    )  
  
    SELECT product_id, 
           position_in_vector,
           value
    FROM vector_products
    WHERE value>0
    """.format(folder1=folder*2,folder2=(folder*2)+1)
    products_dict_test=pass_sql_query_to_get_data_to_train_and_test_and_validation_to_dictonaries_where_the_key_is_the_product_id(path_to_sqlite3,sql)

    return  {
                "train":transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_train),
                "test":transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_test)
            }




def defineBetterModel(path_to_sqlite,data_validation,dropColumnsFromDataTrainAndTest=[]):
    bestModel=None
    averageAcuracyForBestModel=None
    for i in range(7):
        

        data_train_test=get_data_to_train_and_test(path_to_sqlite,i)
        if(len(dropColumnsFromDataTrainAndTest)>0):
            data_train_test['train']['dataframeX']=data_train_test['train']['dataframeX'].drop(dropColumnsFromDataTrainAndTest, axis=1)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(data_train_test['train']['dataframeX'], data_train_test['train']['Y'])
        ypred=clf.predict(data_validation['dataframeX'])

        confusion_mat=confusion_matrix(data_validation['Y'], ypred, labels=pd.Series(list(ypred)+data_validation['Y']).unique())
        avg_acc=0.0
        for i in range(len(confusion_mat)):
            avg_acc=avg_acc+(float(confusion_mat[i,i])/float(sum((confusion_mat[i,:]))))

        avg_acc=avg_acc/float(len(confusion_mat))
        if(averageAcuracyForBestModel== None or avg_acc>averageAcuracyForBestModel):
            averageAcuracyForBestModel=avg_acc
            bestModel=clf

    return {'bestModel':bestModel,'averageAcuracy':averageAcuracyForBestModel}

def create_dataFrame_with_confusion_matrix(real,pred):
    labels=list(pd.Series(list(real)+list(pred)).unique())
    confusion_mat=confusion_matrix(real, pred, labels=labels) 
    
    df=pd.DataFrame(confusion_mat,columns=[['predito']*len(labels),labels], index=[['real']*len(labels),labels])
    return df

#####columns_to_drop=[]
#####data_validation=get_data_to_validation(path_to_sqlite)
#####if(len(columns_to_drop)>0):
#####    data_validation['dataframeX']=data_validation['dataframeX'].drop(columns_to_drop,axis=1)
#####return_from_defineBetterModel=defineBetterModel(path_to_sqlite,data_validation,dropColumnsFromDataTrainAndTest=columns_to_drop)
#####model=return_from_defineBetterModel['bestModel']
#####ypred=model.predict(data_validation['dataframeX'])

#####confusion_mat=create_dataFrame_with_confusion_matrix(data_validation['Y'],ypred)
#####print("Matriz de confusão:")
#####confusion_mat

#####print("Acurácia media: "+str(return_from_defineBetterModel['averageAcuracy']))
#%%[markdown]
# Um fato interessante é que as variávies  price, weight e express_delivery parecem não contribuir para
# na construção modelo, podendo até mesmo atrapalhar na elaboração do mesmo. Tal afirmação pode ser 
# confirmada nos resultados obtidos pelo modelo a seguir, onde estas variáveis de entrada foram retiradas da criação do mesmo:

#%%
#####columns_to_drop=['price', 'express_delivery','weight']
#####data_validation=get_data_to_validation(path_to_sqlite)
#####if(len(columns_to_drop)>0):
#####    data_validation['dataframeX']=data_validation['dataframeX'].drop(columns_to_drop,axis=1)
#####return_from_defineBetterModel=defineBetterModel(path_to_sqlite,data_validation,dropColumnsFromDataTrainAndTest=columns_to_drop)
#####model=return_from_defineBetterModel['bestModel']
#####ypred=model.predict(data_validation['dataframeX'])

#####confusion_mat=create_dataFrame_with_confusion_matrix(data_validation['Y'],ypred)
#####print("Matriz de confusão:")
#####confusion_mat

#####print("Acurácia media: "+str(return_from_defineBetterModel['averageAcuracy']))


#%%[markdown]
## Sistema de termos de busca
# Para esta tarefa vai ser utilizado os histogramas das palavras digitadas nas consultas contudo e 
# os principais algoritmos nessa tarefa vão ser a PCA e o K-médias. Ambos funcionam melhor se tiverem,
# que lidar com apenas algumas dezenas de variaveis, ao invés de centenas, por isso as metricas 
# utilizadas nessa seção vão ser derivadas  de operações entre os histogramas das consultas com os
# histogramas médios das palavras por categoria. O histogra médio de palavras por categoria pode ser 
# montado aplicando a média sobre todos os histogramas dos produtos divididos por categoria.
#
# As operações entre os histogramas que vão utilizas vão ser a o angulo entre vetores e a distância 
# entre vetores. A seguir o código que gera as metricas descritas.

#%%
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
   
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def calculate_histogram_of_words_for_categories(path_to_sqlite3):
    conn = sqlite3.connect(path_to_sqlite3)
    
    sql="""
    
    WITH vector_products AS (
        SELECT query_elo7.product_id,
            vector_element.position_in_vector,
            query_elo7.category,
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id        
  
        GROUP BY query_elo7.product_id,vector_element.position_in_vector,query_elo7.category
        ORDER BY query_elo7.product_id,vector_element.position_in_vector,query_elo7.category
    )  
    
    SELECT category, 
           position_in_vector,
           AVG(value) AS value
    FROM vector_products    
    GROUP BY category,position_in_vector
    ORDER BY category,position_in_vector        
    """
    df=pd.read_sql_query(sql,conn)
    category_vectors={}
    for category in df['category'].unique():
        category_vectors[category]=df[df['category']==category].sort_values(by=['position_in_vector']).values.transpose()[2]

    return category_vectors

def compute_angle_from_categories_hist_and_distance_from_categories_hist_for_query_hist(query_info,category_vectors):
    categories_names=list(category_vectors.keys())
    categories_names.sort()
    correlations=[]
    angles=[]
    for category_name in categories_names:
        correlations.append(np.corrcoef(np.array([category_vectors[category_name],query_info['histogram']],dtype=float))[0,1])
        angles.append(angle_between(category_vectors[category_name],query_info['histogram']))
    return {'correlations':correlations,'angles':angles}



def get_histograms_for_querys(path_to_sqlite3):
    querys_info={}

    sql='''
    WITH TEMP1 AS(
            SELECT vector_element.query_elo7_id ,
            vector_element.position_in_vector,
            AVG(vector_element.value) AS value    
            FROM vector_element       
  
        GROUP BY vector_element.query_elo7_id,vector_element.position_in_vector
        ORDER BY vector_element.query_elo7_id,vector_element.position_in_vector
    )
    SELECT query_elo7_id,
           position_in_vector,
           value
    FROM TEMP1
    WHERE value>0
    ORDER BY query_elo7_id,position_in_vector
    '''
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur.execute("""
        SELECT  query_elo7_id,   
                price,
                weight,
                express_delivery,
                minimum_quantity,
                view_counts
      
        FROM query_elo7;
    """)
    
    for row in cur:
        querys_info[row[0]]={'price':row[1],'weigth':row[2],'express_delivery':row[3],'minimum_quantity':row[4],'view_counts':row[5],'histogram':np.zeros(384),'correlation_from_categories':[], 'angle_from_categories':[]}
    
    cur.execute(sql)
    for row in cur:
        querys_info[row[0]]['histogram'][row[1]]=row[2]

    conn.close()
    
    category_vectors=calculate_histogram_of_words_for_categories(path_to_sqlite3)
    for query_key in querys_info:
        query_info=querys_info[query_key]

        
        angles_and_correlations_for_query=compute_angle_from_categories_hist_and_distance_from_categories_hist_for_query_hist(query_info,category_vectors)
        query_info['correlation_from_categories']=angles_and_correlations_for_query['correlations']
        query_info['angle_from_categories']=angles_and_correlations_for_query['angles']

    return querys_info

def create_dataframe_for_angles_from_categories_and_some_columns_from_table_query_elo7(querys_info):
    data_frame_in_dict={'price':[],'weigth':[],'express_delivery':[],'minimum_quantity':[]}
    numberOfCorrelations=len(querys_info[list(querys_info.keys())[0]]['correlation_from_categories'])
    for i in range(numberOfCorrelations):
        data_frame_in_dict['angle_'+str(i)]=[]

    for key_query in querys_info:
        query=querys_info[key_query]
        for i in range(len(query['angle_from_categories'])):
            angle_from_category=query['angle_from_categories'][i]
            data_frame_in_dict['angle_'+str(i)].append(angle_from_category)
        
        data_frame_in_dict['price'].append(query['price'])
        data_frame_in_dict['weigth'].append(query['weigth'])
        data_frame_in_dict['express_delivery'].append(query['express_delivery'])
        data_frame_in_dict['minimum_quantity'].append(query['minimum_quantity'])
    return pd.DataFrame.from_dict(data_frame_in_dict)

querys_info=get_histograms_for_querys(path_to_sqlite)
df=create_dataframe_for_angles_from_categories_and_some_columns_from_table_query_elo7(querys_info)
print('hello')
#https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
#%% [markdown]
# Outra métrica criada a partir dos histogramas é quão parecido o histograma de um produto
# é do histograma de uma categoria, para saber se dois vetores são parecidos pode-se utilizar
# o cálculo do angulo entre eles, a seguir o código que faz esse cálculo:
#%%



histogram_categories=calculate_histogram_of_words_for_categories(path_to_sqlite)

