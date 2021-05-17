#%%[markdown]
## Carregar base de dados para o SQLite
#Definir localização da base de dados 
#%%
path_to_database='data/raw/elo7_recruitment_dataset.csv'
#%%[markdown]
#Definir localização onde SQLite vai ser guardado, é recomendavel usar uma partição
#mapeada em RAM para aumentar a performance (exemplo /dev/shm)
#%%
path_to_sqlite='/dev/shm/database.sqlite3' #Store the database in ram partition (as /dev/shm) to increase the performance


#%%[markdown]
#Passar os dados de CSV para SQLITE, isso é para facilitar algumas
#análises porque o SQLITE é uma linguagem de consulta de dados que permite
#facíl manipulação dos dados.
#%%
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from IPython.display import display
from sklearn import linear_model
import pickle

#########nltk.download('punkt')
#########nltk.download('stopwords')
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
    df['price_group']=df.apply((lambda row: -1), axis = 1)
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

#########pass_csv_database_to_sqlite3(path_to_sqlite,path_to_database)


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


#########create_schema_for_tables_that_associate_words_in_querys_with_querys_typed(path_to_sqlite)
#########populate_table__word_typed_in_query(path_to_sqlite)
#########create_schema_for_table___word_typed_in_query___query_elo7(path_to_sqlite)
#########word_typed_in_query___query_elo7(path_to_sqlite)
#########create_schema_for_table___vector_element(path_to_sqlite)


#%%[markdown]
## Análise exploratoria
### Análises das palavras dígitadas nas consultas
###### Para estas análises foram retiradas as palavras conhecidas em *PLN* (processamento de linguagem natural) como *stopwords*, *stopwords* são palavras muito comuns que não adicionam significado para o texto (preposições por exemplo)
#Contagem do número de palavras distintas que já foram digitadas por
#usuários em consultadas, admitindo que nos modelos que vão ser desenvolvidos as
#palavras vão ser as variáveis de entrada, estas são as variáveis que primeiro vão ser análisadas.
#%%
#########conn=sqlite3.connect(path_to_sqlite)
#########sql="""SELECT COUNT(DISTINCT word)        
#########        FROM word_typed_in_query
#########        WHERE word NOT IN ({stopwords})
#########    """.format(stopwords=str(list(stopwords.words('portuguese')))[1:-1])
#########print("Número de palavras distintas já digitadas: "+str(conn.execute(sql).fetchall()[0][0]))
#########conn.close()

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

#########print("Análise de frequência das vinte palavras mais digitadas nas consultas:")
#########df_number_of_times_for_words_in_querys=count_number_of_times_that_word_appear_in_query(path_to_sqlite)
#########df_number_of_times_for_words_in_querys.head(20)
# %%[markdown]
# Pode-se notar um decaimento exponencial (muito rápido) na 
# frequencia da palavra mais digitada para vigesima mais digitada. <br>
# Uma melhor análise seria plotar um gráfico para confirmar
# este decaimento exponencial onde, o eixo X representa o
# *ranking* da palavra entre as que mais aparecem e o eixo Y o número de 
# vezes que ela aparece.

# %%
#########df_number_of_times_for_words_in_querys=count_number_of_times_that_word_appear_in_query(path_to_sqlite)
#########df_number_of_times_for_words_in_querys=df_number_of_times_for_words_in_querys.reset_index()
#########df_number_of_times_for_words_in_querys.rename(columns = {'index':'ranking da palavra', 'numbero_de_consultas_onde_a_palavra_foi_digitada':'número de vezes que aparece'}, inplace = True)
#########sns.lineplot(data=df_number_of_times_for_words_in_querys.reset_index(), x="ranking da palavra", y="número de vezes que aparece")

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
# pelo menos uma palavra do grupo que esta dentro da consulta".<br> <br>
# A seguir um plot que mostra quantas consultas são cobertas por grupos com 
# as N palavras mais frequentes:

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

#########df_with_number_of_queries_coverage_by_groups_with_the_N_most_frequent_words=number_of_queries_coverage_by_groups_with_the_N_most_frequent_words(path_to_sqlite,path_to_database)
#########sns.lineplot(data=df_with_number_of_queries_coverage_by_groups_with_the_N_most_frequent_words, x="grupo com as N palavras mais frequentes", y="número de consultas cobertas pelo grupo")

# %%
#########last_row_for_infomation_about_group_of_words=(df_with_number_of_queries_coverage_by_groups_with_the_N_most_frequent_words.values)[-1]
#########print ("Quantidade consultas cobertas pelo grupo com as {num_of_words} palavras mais frequentes: {num_of_querys}".format(num_of_words=last_row_for_infomation_about_group_of_words[0],num_of_querys=last_row_for_infomation_about_group_of_words[1]))



# %%[markdown]
# Observando o gráfico anterior observa-se que as 384 das palavras mais frequentes
# estão em 35591 das 38507 consultas que estão disponiveis na base dados fornecida,
# aproximadamente 92% de cobertura. Deste modo os modelos de IA desenvolvidos caso 
# recebam uma consulta/query como uma de suas entradas vão analisa apenas estas 384 
# palavras.





# %%[markdown]
### Análises dos atributos dos produtos existentes (peso, preço e entrega rápida).



#### A seguir histogramas para diferentes faixas  peso que os produtos possuem em cada categoria:
# %%[markdown]
# Histogramas com a distribuição de peso por categoria.
#%%
#########conn=sqlite3.connect(path_to_sqlite)

#########df=pd.read_sql_query("""SELECT DISTINCT product_id, category, weight FROM query_elo7 WHERE weight""",conn)
#########sns.histplot(hue="category", x="weight", data=df,bins=10)
#########conn.close()
# %%
# %%[markdown]
# Histogramas com a distribuição de peso por categoria com limite de peso de até 40, isso é equivalente a um zoom no inicio no eixo X dos histogramas do gráfico anterior.
#%%
#########conn=sqlite3.connect(path_to_sqlite)

#########df=pd.read_sql_query("""SELECT DISTINCT product_id, category, weight FROM query_elo7 WHERE weight<40""",conn)

#########sns.histplot(hue="category", x="weight", data=df,bins=10)
#########conn.close()


# %%[markdown]
#### A seguir histogramas para diferentes faixas preço que os produtos possuem em cada categoria:
# %%[markdown]
# Histogramas com a distribuição de preço por categoria.
#%%
#########conn=sqlite3.connect(path_to_sqlite)

#########df=pd.read_sql_query("""SELECT DISTINCT product_id, category, price FROM query_elo7 WHERE price""",conn)
#########sns.histplot(hue="category", x="price", data=df,bins=10)
#########conn.close()

# %%[markdown]
# Histogramas com a distribuição de peso por categoria com limite de preço de até 100, isso é equivalente a um zoom no inicio do eixo X dos histogramas do gráfico anterior.
#%%
#########conn=sqlite3.connect(path_to_sqlite)

#########df=pd.read_sql_query("""SELECT DISTINCT product_id, category, price FROM query_elo7 WHERE price<200  """,conn)
#ax = sns.boxplot(x="category", y="weight", data=df)
#########sns.histplot(hue="category", x="price", data=df,bins=10)
#########conn.close()


# %%[markdown]
# Nos histogramas do gráfico anterior a categoria "Lembrancinhas" esta atrapalhando 
# a visualizaçao das distribuição de preços das outras categorias, ela vai ser retirada
# do próximo gráfico então. O gráifco a seguir é um replot dos histogramas do gráfico 
# anterior porem sem a categoria "Lembrancinhas":

#%%
#########conn=sqlite3.connect(path_to_sqlite)

#########df=pd.read_sql_query("""SELECT DISTINCT product_id, category, price FROM query_elo7 WHERE  price<200 AND category!='Lembrancinhas'  """,conn)
#ax = sns.boxplot(x="category", y="weight", data=df)
#########sns.histplot(hue="category", x="price", data=df,bins=10)
#########conn.close()


# %%[markdown]
#### Análise sobre a distribuição de peso e preço 
# Como pode ser observado nem o preço nem o peso  dos produtos segue uma distribuição 
# normal portanto, a melhores estrategias para discretizar estes valores seriam:
# <ul>
#  <li>Discretizar os valores resultantes de uma transformação logaritma ou .</li>
#   <li>Clusteriza-los via algum algoritmo de aprendizado de aprendizado não supervisionado como o K-means.</li>
# </ul>
#  É importante pensar em tecnicas para discretizar de valores continuos porque isso pode melhorar
#  qualidade dos modelos de IA desenvolvidos.

# %%[markdown]
#### Análise sobre a distribuição do atributo envio expresso

#%%

#########conn=sqlite3.connect(path_to_sqlite)
#########df=pd.read_sql_query("""SELECT DISTINCT product_id, category, express_delivery FROM query_elo7 WHERE  price<200 AND category!='Lembrancinhas'  """,conn)


#########df['express_delivery']=df['express_delivery'].apply(lambda x: 'yes' if x>0.0 else 'no')
#########categories=df['category'].unique()
#########for category in categories:
#########    print("Distribuição para a categoria:"+str(category)+"\n")
#########    dfT=df[df['category']==category]
#########    sns.histplot( x="express_delivery", data=dfT.sort_values(by=['express_delivery']), stat='probability',discrete=True, shrink=.8)
#########    plt.show()
#########    print("\n\n\n\n")


#########conn.close()
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
#</ul> <br> <br>
# A seguir o código que cria os histogramas para cada consulta:

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

#########populate_table____vector_element(path_to_sqlite)





#%%[markdown]
#### Variáveis utilizadas para a criação do sistema
# Para a criação de sistema foram utilizados os campos price, weight e express_delivery das consultas  
# e o histograma de palavras dos produtos associados as palavras digitadas as buscas.
#### Encontrar o melhor modelo de predição de categorias com base na metodologia K-Fold 
# A maioria dos passos a seguir vão ser repetidos várias vezes com o objetivo de encontrar o melho modelo, esta 
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
# alcançadas para classificar cada categoria. <br><br>

# A seguir a matriz de confusão e a acurácia média para o melhor de modelo de predição encontrado para a elaboração do sistema:
#%%

def transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(database,column_to_be_y='category',fields_to_drop=[]):
    fields=list(database[list(database.keys())[0]].keys())
    dataFrameInDict={}
    for field in fields:
        if(field !="vector"):
            dataFrameInDict[field]=[]

    dataFrameInDict['Y']=[]

    for i in range(384):
        dataFrameInDict['vector_e'+str(i)]=[]

    for data_id in database:
        data=database[data_id]
        for field in fields:
            if(field!="vector"):
                dataFrameInDict[field].append(data[field])
            
        for i in range(384):
            dataFrameInDict['vector_e'+str(i)].append(data['vector'][i])

        dataFrameInDict['Y'].append(data[column_to_be_y])

    df=pd.DataFrame.from_dict(dataFrameInDict)
    df=df.dropna()
    y=df['Y'].values
    df=df.drop(['Y'],axis=1)
    #df['weight'].fillna(value=df['weight'].mean(), inplace=True)
    #df['price'].fillna(value=df['price'].mean(), inplace=True)
    df=df.drop([column_to_be_y],axis=1)
    if(len(fields_to_drop)>0):
        df=df.drop(fields_to_drop,axis=1)
        
    return {'dataframeX':df,'Y':y}


def merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,
                                                                                             query_to_get_words_histograms,
                                                                                             query_to_get_varied_information):
    #Important Note=========================================
    # The first field of query_to_get_words_histograms 
    # and query_to_get_varied_information have to be  
    # equals and they are the for the dictonaries
    #End====================================================
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    
    
    information_about_products={}

    cur.execute(query_to_get_varied_information)

    fields=[]    
    
    for field in cur.description:
        fields.append(field[0])

    for row in cur:
        dictTemp={}
        keyForDict=fields[0]
        for i in range(len(fields)-1):            
            dictTemp[fields[i+1]]=row[i+1]


        information_about_products[row[0]]=dictTemp

    cur.execute(query_to_get_words_histograms)
    productsDict={}

    for row in cur:
        if(not (row[0]  in productsDict)):
            dictTemp={'vector':np.zeros(384)}
            for i in range(len(fields)-1):
                dictTemp[fields[i+1]]=information_about_products[row[0]][fields[i+1]]

            productsDict[row[0]]=dictTemp

        productsDict[row[0]]['vector'][row[1]]=row[2]
    conn.close()
    return productsDict

def get_data_to_validation_to_use_in_the_model_to_predict_categories(path_to_sqlite3):
    query_to_get_words_histograms="""
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
    
    query_to_get_varied_information="""
    SELECT DISTINCT product_id,       
           category,
           weight,
           price,
           express_delivery
    FROM query_elo7;
    """

    products_dict_validation=merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,query_to_get_words_histograms,query_to_get_varied_information)
    return transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_validation)

def get_data_to_train_and_test_to_use_in_the_model_to_predict_categories(path_to_sqlite3,folder):
    query_to_get_varied_information="""
    SELECT DISTINCT product_id,       
           category,
           weight,
           price,
           express_delivery
    FROM query_elo7;
    """

    query_to_get_words_histograms_to_train="""
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
        

    products_dict_train=merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,query_to_get_words_histograms_to_train,query_to_get_varied_information)

    query_to_get_words_histograms_to_test="""
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
    
    products_dict_test=merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,query_to_get_words_histograms_to_test,query_to_get_varied_information)

    return  {
                "train":transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_train),
                "test":transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_test)
            }




def define_better_model_to_predict_categories(path_to_sqlite,data_validation,dropColumnsFromDataTrainAndTest=[]):
    bestModel=None
    averageAcuracyForBestModel=None
    for i in range(7):
        

        data_train_test=get_data_to_train_and_test_to_use_in_the_model_to_predict_categories(path_to_sqlite,i)
        if(len(dropColumnsFromDataTrainAndTest)>0):
            data_train_test['train']['dataframeX']=data_train_test['train']['dataframeX'].drop(dropColumnsFromDataTrainAndTest, axis=1)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(data_train_test['train']['dataframeX'], data_train_test['train']['Y'])
        ypred=clf.predict(data_validation['dataframeX'])

        confusion_mat=confusion_matrix(data_validation['Y'], ypred, labels=pd.Series(list(ypred)+list(data_validation['Y'])).unique())
        avg_acc=0.0
        for i in range(len(confusion_mat)):
            avg_acc=avg_acc+(float(confusion_mat[i,i])/float(sum((confusion_mat[i,:]))))

        avg_acc=avg_acc/float(len(confusion_mat))
        if(averageAcuracyForBestModel== None or avg_acc>averageAcuracyForBestModel):
            averageAcuracyForBestModel=avg_acc
            bestModel=clf

    return {'bestModel':bestModel,'averageAcuracy':averageAcuracyForBestModel}

def create_dataFrame_with_confusion_matrix(real,pred, labels_arg=None):
    if(labels_arg==None):
        labels=list(pd.Series(list(real)+list(pred)).unique())
    else:
        labels=labels_arg
    confusion_mat=confusion_matrix(real, pred, labels=labels) 
    
    df=pd.DataFrame(confusion_mat,columns=[['predito']*len(labels),labels], index=[['real']*len(labels),labels])
    return df

#########columns_to_drop=[]
#########data_validation=get_data_to_validation_to_use_in_the_model_to_predict_categories(path_to_sqlite)
#########if(len(columns_to_drop)>0):
#########    data_validation['dataframeX']=data_validation['dataframeX'].drop(columns_to_drop,axis=1)
#########return_from_defineBetterModel=define_better_model_to_predict_categories(path_to_sqlite,data_validation,dropColumnsFromDataTrainAndTest=columns_to_drop)
#########model=return_from_defineBetterModel['bestModel']
#########ypred=model.predict(data_validation['dataframeX'])

#########confusion_mat_0=create_dataFrame_with_confusion_matrix(data_validation['Y'],ypred)
#########print("Matriz de confusão para classificar produtos:")
#########display(confusion_mat_0)

#########print("Acurácia media: "+str(return_from_defineBetterModel['averageAcuracy']))
#%%[markdown]
# Um fato interessante é que as variávies  price, weight e express_delivery parecem não contribuir para
# na construção modelo, podendo até mesmo atrapalhar na elaboração do mesmo. Tal afirmação pode ser 
# confirmada nos resultados obtidos pelo modelo a seguir, onde estas variáveis de entrada foram retiradas da criação do mesmo:

#%%
#########columns_to_drop=['price', 'express_delivery','weight']
#########data_validation=get_data_to_validation_to_use_in_the_model_to_predict_categories(path_to_sqlite)
#########if(len(columns_to_drop)>0):
#########    data_validation['dataframeX']=data_validation['dataframeX'].drop(columns_to_drop,axis=1)
#########return_from_defineBetterModel=define_better_model_to_predict_categories(path_to_sqlite,data_validation,dropColumnsFromDataTrainAndTest=columns_to_drop)
#########model=return_from_defineBetterModel['bestModel']
#########ypred=model.predict(data_validation['dataframeX'])

#########confusion_mat_1=create_dataFrame_with_confusion_matrix(data_validation['Y'],ypred)
#########print("Matriz de confusão para classificar produtos apenas com os histogramas:")
#########display(confusion_mat_1)

#########print("Acurácia media: "+str(return_from_defineBetterModel['averageAcuracy']))

#########model_predict_category_fp=open( "data/models/predict_category/mode.pickle", "wb" )
#########pickle.dump( model, model_predict_category_fp)
#########model_predict_category_fp.close()

#%%[markdown]
## Sistema de termos de busca

# Para criar este sistema foi primeiro pensado na ideia de criar grupos de preço e 
# a partir destes grupos utilizar um algoritmo de  aprendizado de maquina que 
# encontre uma relação entre os histogramas de palavras das consultas
# com um determinado grupo de preço, descobrindo assim a intenção do usuário 
# de comprar produtos de um terminado preço com base naquilo que digita em suas
# consultas. 
# Para a criação dos grupo de preço foi  utilizado o algoritmo K-médias.  A seguir
# o código que divide os preços em grupos:


#%%
def create_discretization_model_from_k_means_for_product_price(path_to_sqlite3,n_clusters):
    conn = sqlite3.connect(path_to_sqlite3)
    sql="""
    SELECT query_elo7_id, price, minimum_quantity
    FROM query_elo7;
    """
    df=pd.read_sql_query(sql,conn)
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)

    df['price_log']=df['price'].apply(np.log)
    #df['price_log']=df['price']
    cluster_labels = clusterer.fit_predict(df['price_log'].values.reshape(-1, 1) )
    silhouette_avg = silhouette_score(df['price_log'].values.reshape(-1, 1), cluster_labels)
    conn.close()
    return {"model":clusterer,'score':silhouette_avg}

#%% [markdown]
# Para computar o número ideal de valores discretos que a variável price deve ter vai ser
# utilizado o metódo da silueta média, onde quanto menor melhor, a seguir os valores obtidos
# pelo metodo da silueta quando price e discretizado para 2,3,5,8,13 e 21 valores distintos.

#%%

#########for i in [2,3,5,8,13,21]:
#########    score=create_discretization_model_from_k_means_for_product_price(path_to_sqlite,i)['score']
#########    print('Discretizando price para {i} valores distintos o valor da silueta média é {resu}.'.format(i=i,resu=score))

#Use Kmeans with 13 clusters to discretize price==============================
def update_column_price_group_with_groups_found_in_kmeans_for_price(path_to_sqlite):
    conn = sqlite3.connect(path_to_sqlite)
    
    model=create_discretization_model_from_k_means_for_product_price(path_to_sqlite,8)['model']
    df=pd.read_sql_query("SELECT query_elo7_id,price, minimum_quantity FROM query_elo7;",conn)
    df['price_log']=df['price'].apply(np.log)    
    df['price_group']=df['price_log'].apply(lambda x: model.predict([[x]])[0])
    cur=conn.cursor()
    progress=0
    for index, row in df.iterrows():
        cur.execute("UPDATE query_elo7 SET price_group={group} WHERE query_elo7_id={query_elo7_id}".format(group=row['price_group'],query_elo7_id=row['query_elo7_id']))
        progress=progress+1

    conn.commit()
    conn.close()


#update_column_price_group_with_groups_found_in_kmeans_for_price(path_to_sqlite)
#end===========================================================================


#%%[markdown]
#Como pode ser observado  a divisão da variável price em oito grupos apresenta um bom resultado 
#para o metódo da silueta. A seguir um dataframe com os valores mínimo e máximo que cada um dos 13
#grupos da variável price apresenta:

#########conn=sqlite3.connect(path_to_sqlite)
#########df=pd.read_sql_query("""SELECT  price_group,
#########                                MIN(price),
#########                                MAX(price)
#########                        FROM query_elo7
#########                        GROUP BY price_group
#########                        ORDER BY MIN(price),MAX(price)
#########                        """,conn)
#########conn.close()
#########display(df)



#%%[markdown]
### Modelo para predição da intenção de preço do usuario por árvore de decisão
# A seguir um modelo que infere a intenção de preço do usuário,
# assim como o modelo que categoriza produtos ele foi criado via
#árvore de decisão e utilizando a metodologia K-fold:


#%%


def get_data_to_validation_to_use_in_the_model_to_predict_price(path_to_sqlite3):
    query_to_get_words_histograms="""
    WITH vector_query AS (
        SELECT query_elo7.query_elo7_id,
            vector_element.position_in_vector,       
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id
        WHERE query_elo7.hash_for_query_elo7_id>=15
  
        GROUP BY query_elo7.query_elo7_id,vector_element.position_in_vector
        ORDER BY query_elo7.query_elo7_id,vector_element.position_in_vector
    )  
  
    SELECT query_elo7_id, 
           position_in_vector,
           value
    FROM vector_query
    WHERE value>0
    """
    
    query_to_get_varied_information="""
    SELECT query_elo7_id,                 
           price_group
    FROM query_elo7;
    """

    products_dict_validation=merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,query_to_get_words_histograms,query_to_get_varied_information)
    return transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_validation, column_to_be_y='price_group')


def get_data_to_validation_to_use_in_the_model_to_regress_price(path_to_sqlite3):
    query_to_get_words_histograms="""
    WITH vector_query AS (
        SELECT query_elo7.query_elo7_id,
            vector_element.position_in_vector,       
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id
        WHERE query_elo7.hash_for_query_elo7_id>=15
  
        GROUP BY query_elo7.query_elo7_id,vector_element.position_in_vector
        ORDER BY query_elo7.query_elo7_id,vector_element.position_in_vector
    )  
  
    SELECT query_elo7_id, 
           position_in_vector,
           value
    FROM vector_query
    WHERE value>0
    """
    
    query_to_get_varied_information="""
    SELECT query_elo7_id,                 
           price
    FROM query_elo7;
    """

    products_dict_validation=merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,query_to_get_words_histograms,query_to_get_varied_information)
    return transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_validation, column_to_be_y='price')




def get_data_to_train_and_test_to_use_in_the_model_to_predict_prices(path_to_sqlite3,folder):

    query_to_get_varied_information="""
    SELECT query_elo7_id,                 
           price_group
    FROM query_elo7;
    """

    query_to_get_words_histograms_to_train="""
    WITH vector_query AS (
        SELECT query_elo7.query_elo7_id,
            vector_element.position_in_vector,       
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id
        WHERE query_elo7.hash_for_query_elo7_id<15 AND query_elo7.hash_for_query_elo7_id  NOT IN ({folder1},{folder2})
  
        GROUP BY query_elo7.query_elo7_id,vector_element.position_in_vector
        ORDER BY query_elo7.query_elo7_id,vector_element.position_in_vector
    )  
  
    SELECT query_elo7_id, 
           position_in_vector,
           value
    FROM vector_query
    WHERE value>0
    """.format(folder1=folder*2,folder2=(folder*2)+1)
    
    products_dict_train=merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,query_to_get_words_histograms_to_train,query_to_get_varied_information)



    query_to_get_words_histograms_to_test="""
    WITH vector_query AS (
        SELECT query_elo7.query_elo7_id,
            vector_element.position_in_vector,       
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id
        WHERE query_elo7.hash_for_query_elo7_id<15 AND query_elo7.hash_for_query_elo7_id   IN ({folder1},{folder2})
  
        GROUP BY query_elo7.query_elo7_id,vector_element.position_in_vector
        ORDER BY query_elo7.query_elo7_id,vector_element.position_in_vector
    )  
  
    SELECT query_elo7_id, 
           position_in_vector,
           value
    FROM vector_query
    WHERE value>0
    """.format(folder1=folder*2,folder2=(folder*2)+1)

    
    products_dict_test=merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,query_to_get_words_histograms_to_test,query_to_get_varied_information)

    return  {
                "train":transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_train, column_to_be_y='price_group'),
                "test":transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_test, column_to_be_y='price_group')
            }



def get_data_to_train_and_test_to_use_in_the_model_to_regress_prices(path_to_sqlite3,folder):

    query_to_get_varied_information="""
    SELECT query_elo7_id,                 
           price
    FROM query_elo7;
    """

    query_to_get_words_histograms_to_train="""
    WITH vector_query AS (
        SELECT query_elo7.query_elo7_id,
            vector_element.position_in_vector,       
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id
        WHERE query_elo7.hash_for_query_elo7_id<15 AND query_elo7.hash_for_query_elo7_id  NOT IN ({folder1},{folder2})
  
        GROUP BY query_elo7.query_elo7_id,vector_element.position_in_vector
        ORDER BY query_elo7.query_elo7_id,vector_element.position_in_vector
    )  
  
    SELECT query_elo7_id, 
           position_in_vector,
           value
    FROM vector_query
    WHERE value>0
    """.format(folder1=folder*2,folder2=(folder*2)+1)
    
    products_dict_train=merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,query_to_get_words_histograms_to_train,query_to_get_varied_information)



    query_to_get_words_histograms_to_test="""
    WITH vector_query AS (
        SELECT query_elo7.query_elo7_id,
            vector_element.position_in_vector,       
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id
        WHERE query_elo7.hash_for_query_elo7_id<15 AND query_elo7.hash_for_query_elo7_id   IN ({folder1},{folder2})
  
        GROUP BY query_elo7.query_elo7_id,vector_element.position_in_vector
        ORDER BY query_elo7.query_elo7_id,vector_element.position_in_vector
    )  
  
    SELECT query_elo7_id, 
           position_in_vector,
           value
    FROM vector_query
    WHERE value>0
    """.format(folder1=folder*2,folder2=(folder*2)+1)

    
    products_dict_test=merge_query_to_get_words_histograms_and_query_to_get_varied_information_into_dictonaries(path_to_sqlite3,query_to_get_words_histograms_to_test,query_to_get_varied_information)

    return  {
                "train":transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_train, column_to_be_y='price'),
                "test":transform_products_load_as_dict_to_dataframeX_variable_to_fit_Y(products_dict_test, column_to_be_y='price')
            }



def define_better_model_to_predict_price_from_text_in_query(path_to_sqlite,data_validation,dropColumnsFromDataTrainAndTest=[]):
    bestModel=None
    averageAcuracyForBestModel=None
    for i in range(7):
        

        data_train_test=get_data_to_train_and_test_to_use_in_the_model_to_predict_prices(path_to_sqlite,i)
        if(len(dropColumnsFromDataTrainAndTest)>0):
            data_train_test['train']['dataframeX']=data_train_test['train']['dataframeX'].drop(dropColumnsFromDataTrainAndTest, axis=1)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(data_train_test['train']['dataframeX'], data_train_test['train']['Y'])
        ypred=clf.predict(data_validation['dataframeX'])

        confusion_mat=confusion_matrix(data_validation['Y'], ypred, labels=pd.Series(list(ypred)+data_validation['Y']).unique())
        avg_acc=0.0
        for i in range(len(confusion_mat)):
            try:
                avg_acc=avg_acc+(float(confusion_mat[i,i])/float(sum((confusion_mat[i,:]))))
            except:
                pass

        avg_acc=avg_acc/float(len(confusion_mat))
        if(averageAcuracyForBestModel== None or avg_acc>averageAcuracyForBestModel):
            averageAcuracyForBestModel=avg_acc
            bestModel=clf

    return {'bestModel':bestModel,'averageAcuracy':averageAcuracyForBestModel}

def create_a_list_with_range_of_values_from_list_with_price_group(path_to_sqlite,list_with_price_group):
    conn=sqlite3.connect(path_to_sqlite)
    df=pd.read_sql_query("""SELECT  price_group,
                                MIN(price),
                                MAX(price)
                        FROM query_elo7
                        GROUP BY price_group
                        ORDER BY MIN(price),MAX(price)
                        """,conn).values
    conn.close()
    conv_group_number_to_range_of_values={}
    for line in df:
        conv_group_number_to_range_of_values[line[0]]=("%.2f"%line[1])+"_to_"+("%.2f"%line[2])
    
    list_with_range_of_values=[]
    for x in list_with_price_group:
        list_with_range_of_values.append(conv_group_number_to_range_of_values[x])
    return list_with_range_of_values

def get_ordination_to_show_value_ranges(path_to_sqlite):
    conn=sqlite3.connect(path_to_sqlite)
    df=pd.read_sql_query("""SELECT  price_group,
                                MIN(price),
                                MAX(price)
                        FROM query_elo7
                        GROUP BY price_group
                        ORDER BY MIN(price),MAX(price)
                        """,conn).values
    conn.close()
    ordination=[]
    for line in df:
        ordination.append(("%.2f"%line[1])+"_to_"+("%.2f"%line[2]))
    return ordination
    
data_validation=get_data_to_validation_to_use_in_the_model_to_predict_price(path_to_sqlite)
return_from_defineBetterModel=define_better_model_to_predict_price_from_text_in_query(path_to_sqlite,data_validation)
#########model=return_from_defineBetterModel['bestModel']
#########ypred=model.predict(data_validation['dataframeX'])

#########data_validation['Y']=create_a_list_with_range_of_values_from_list_with_price_group(path_to_sqlite,data_validation['Y'])
#########ypred=create_a_list_with_range_of_values_from_list_with_price_group(path_to_sqlite,ypred)
#########ordination_to_show_value_ranges=get_ordination_to_show_value_ranges(path_to_sqlite)
#########confusion_mat_2=create_dataFrame_with_confusion_matrix(data_validation['Y'],ypred,labels_arg=ordination_to_show_value_ranges)
#%% [markdown]
# A seguir a matriz de confusão é a acuracia média que o modelo conseguiu para diferentes
# grupos de preço:
#%%
#########print("Matriz de confusão para classifcar grupos de preço:")
#########display(confusion_mat_2)
#########print("Acurácia media: "+str(return_from_defineBetterModel['averageAcuracy']))

#%%[markdown]
# Como pode ser observado o modelo de predição desenvolvido não apresentou um 
# bom resultado. Isto indica que o talvez a discretização dos preços por faixas 
# utilizando K-médias não seja a melhor alternativa. Duas ideias foram pensadas
# para criar um algoritmo que faça uma melhor separação dos preços com base no
# texto de consulta digitado:
# <ul>
# <li> 
#  Calcular o ângulo  entre os histogramas de palavras das consultas
#  com os histogramas médios de palavras por categoria de preço. Isso iria gerar 8 atributos
#  númericos para cada consulta e por meio da PCA eles seriam reduzidos para 2 para assim,
#  , ser possível gerar um *scater plot* e analisar se existe uma separação
#  natural de  *clusters* de intenção de preço.
#  </li>
# <li> Aplicar uma transformação logaritma sobre os preços e utilizar algum 
# algoritmo de regessão (redes neurais ou minímos quadrados) para criar um
# modelo de predição de preços. A entrada do modelo seriam os 8 atributos 
# gerados pela ideia anterior porque, isso diminuiria a dimensionalidade 
# do problema o que faria o modelo gerado ter mais change de ser preciso.</li>
# </ul> <br>
# ## Análise de PCA
# A seguir a esta código que implementa a primeira ideia proposta antes:
#%%
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


def calculate_histogram_of_words_for_price_groups(path_to_sqlite3,where_clausule=""):
    conn = sqlite3.connect(path_to_sqlite3)
    
    sql="""
    
    WITH vector_products AS (
        SELECT query_elo7.query_elo7_id,
            vector_element.position_in_vector,
            query_elo7.price_group,
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id        
        {WHERE}
        GROUP BY query_elo7.query_elo7_id,vector_element.position_in_vector,query_elo7.price_group
        ORDER BY query_elo7.query_elo7_id,vector_element.position_in_vector,query_elo7.price_group
    )  
    
    SELECT price_group, 
           position_in_vector,
           AVG(value) AS value
    FROM vector_products    
    GROUP BY price_group,position_in_vector
    ORDER BY price_group,position_in_vector        
    """.format(WHERE=where_clausule)
    df=pd.read_sql_query(sql,conn)
    price_group_vectors={}
    for price_group in df['price_group'].unique():
        price_group_vectors[price_group]=df[df['price_group']==price_group].sort_values(by=['position_in_vector']).values.transpose()[2]

    #code for debug purpose===========================================================================
    sql=""" 
        SELECT query_elo7.query_elo7_id,
            vector_element.position_in_vector,
            query_elo7.price_group,
            SUM(vector_element.value) AS value    
            FROM vector_element
        INNER JOIN query_elo7 ON query_elo7.query_elo7_id=vector_element.query_elo7_id        
        {WHERE}
        GROUP BY query_elo7.query_elo7_id,vector_element.position_in_vector,query_elo7.price_group
        ORDER BY query_elo7.query_elo7_id,vector_element.position_in_vector,query_elo7.price_group""".format(WHERE=where_clausule)
    df=pd.read_sql_query(sql,conn)
    #end==============================================================================================
    
    conn.close()
    return {'price_group_vectors':price_group_vectors,'query_elo7_id_used':df['query_elo7_id'].values}

def compute_angle_from_some_grouping_hist_and_distance_from_small_group_hist_for_query_hist(query_info,category_vectors):
    categories_names=list(category_vectors.keys())
    categories_names.sort()
    correlations=[]
    angles=[]
    for category_name in categories_names:
        correlations.append(np.corrcoef(np.array([category_vectors[category_name],query_info['histogram']],dtype=float))[0,1])
        angles.append(angle_between(category_vectors[category_name],query_info['histogram']))
    return {'correlations':correlations,'angles':angles}



def get_histograms_for_querys(path_to_sqlite3,
                              where_clausule_for_vector_element="",
                              where_clausule_for_query_elo7="",
                              where_for_calculate_histogram_of_words_for_price_groups=""):
    querys_info={}

    sql='''
    WITH TEMP1 AS(
            SELECT vector_element.query_elo7_id ,
            vector_element.position_in_vector,
            AVG(vector_element.value) AS value    
            FROM vector_element               
            {WHERE}
        GROUP BY vector_element.query_elo7_id,vector_element.position_in_vector
        ORDER BY vector_element.query_elo7_id,vector_element.position_in_vector
    )
    SELECT query_elo7_id,
           position_in_vector,
           value
    FROM TEMP1
    WHERE value>0    
    ORDER BY query_elo7_id,position_in_vector
    '''.format(WHERE=where_clausule_for_vector_element)
    conn = sqlite3.connect(path_to_sqlite3)
    cur=conn.cursor()
    cur.execute("""
        SELECT  query_elo7_id,   
                price,
                weight,
                express_delivery,
                minimum_quantity,
                view_counts
      
        FROM query_elo7
        {WHERE};
    """.format(WHERE=where_clausule_for_query_elo7))
    
    for row in cur:
        querys_info[row[0]]={'query_elo7_id':row[0],'price':row[1],'weigth':row[2],'express_delivery':row[3],'minimum_quantity':row[4],'view_counts':row[5],'histogram':np.zeros(384),'correlation_from_categories':[], 'angle_from_categories':[]}
    
    cur.execute(sql)
    for row in cur:
        querys_info[row[0]]['histogram'][row[1]]=row[2]

    conn.close()
    
    ret_from_calculate_histogram_of_words_for_price_groups=calculate_histogram_of_words_for_price_groups(path_to_sqlite3,where_for_calculate_histogram_of_words_for_price_groups)
    group_vectors=ret_from_calculate_histogram_of_words_for_price_groups['price_group_vectors']
    querys_used_for_create_histogram_from_group_of_vectors=ret_from_calculate_histogram_of_words_for_price_groups['query_elo7_id_used']
    for query_key in querys_info:
        query_info=querys_info[query_key]

        
        angles_and_correlations_for_query=compute_angle_from_some_grouping_hist_and_distance_from_small_group_hist_for_query_hist(query_info,group_vectors)
        query_info['correlation_from_price_groups']=angles_and_correlations_for_query['correlations']
        query_info['angle_from_price_groups']=angles_and_correlations_for_query['angles']

    return {'querys_info':querys_info, 'querys_used_for_create_histgram_of_group_of_vectors':querys_used_for_create_histogram_from_group_of_vectors,'group_vectors':group_vectors}





def create_dataframe_for_angles_from_small_group_and_some_columns_from_table_query_elo7(querys_info, small_group_name='correlation_from_categories'):
    data_frame_in_dict={'price':[],'weigth':[],'express_delivery':[],'minimum_quantity':[]}
    numberOfCorrelations=len(querys_info[list(querys_info.keys())[0]]['angle_from_'+small_group_name])
    for i in range(numberOfCorrelations):
        data_frame_in_dict['angle_'+str(i)]=[]

    for key_query in querys_info:
        query=querys_info[key_query]
        for i in range(len(query['angle_from_'+small_group_name])):
            angle_from_category=query['angle_from_'+small_group_name][i]
            data_frame_in_dict['angle_'+str(i)].append(angle_from_category)
        
        data_frame_in_dict['price'].append(query['price'])
        data_frame_in_dict['weigth'].append(query['weigth'])
        data_frame_in_dict['express_delivery'].append(query['express_delivery'])
        data_frame_in_dict['minimum_quantity'].append(query['minimum_quantity'])
    return pd.DataFrame.from_dict(data_frame_in_dict)




#########querys_info=get_histograms_for_querys(path_to_sqlite)['querys_info']
#########df=create_dataframe_for_angles_from_small_group_and_some_columns_from_table_query_elo7(querys_info,'price_groups')
#########df=df.dropna()
#########df['price']=df['price'].apply(np.log)
#########x = df.loc[:, [ 'angle_0','angle_1','angle_2','angle_3','angle_4','angle_5']].values

#########y = df.loc[:,['price']].values
#########x = StandardScaler().fit_transform(x)
#########pca = PCA(n_components=2)
#########principalComponents = pca.fit_transform(x)
#########principalDf = pd.DataFrame(data = principalComponents
#########             , columns = ['principal component 1', 'principal component 2'])

#########finalDf = pd.concat([principalDf, df[['price']]], axis = 1)
#########sns.scatterplot(data=finalDf, x="principal component 1", y="principal component 2", hue="price")
#########plt.show()            

#%% [markdown]
# Como pode ser observado pelo *plot* acima parece não haver uma separação muito evidente
# para os diferentes preços sobre as duas primeiras componentes da PCA.

#%%[markdown]
### Modelo para predição da intenção de preço do usuario por regressão linear
# A seguir um modelo que infere a intenção de preço do usuário,
# assim como o modelo que categoriza produtos ele foi criado via
# árvore de decisão e utilizando a metodologia K-fold contudo, como o primeiro fold
# testado no conjunto de validação já obteve um bom resultado outros folds não foram
# analisados:

#%%
def get_histograms_for_querys_from_validation(path_to_sqlite3, folder_of_train_and_test):
    folder=folder_of_train_and_test
    return get_histograms_for_querys(   path_to_sqlite,
                                        "WHERE vector_element.hash_for_query_elo7_id>=15",
                                        "WHERE query_elo7.hash_for_query_elo7_id>=15",
                                        """WHERE vector_element.hash_for_query_elo7_id<15 AND
                                           vector_element.hash_for_query_elo7_id NOT IN ({folder1},{folder2})""".format(folder1=(folder*2),folder2=((folder*2)+1)))
        
def get_histograms_for_querys_for_train_and_test(path_to_sqlite3,folder):
        train=get_histograms_for_querys(    path_to_sqlite,
                                            """WHERE vector_element.hash_for_query_elo7_id<15 AND 
                                                     vector_element.hash_for_query_elo7_id NOT IN ({folder1},{folder2})""".format(folder1=(folder*2),folder2=((folder*2)+1)),
                                            """WHERE query_elo7.hash_for_query_elo7_id<15 AND
                                                     query_elo7.hash_for_query_elo7_id NOT IN ({folder1},{folder2})""".format(folder1=(folder*2),folder2=((folder*2)+1)),
                                            """WHERE vector_element.hash_for_query_elo7_id<15 AND
                                                     vector_element.hash_for_query_elo7_id NOT IN ({folder1},{folder2})""".format(folder1=(folder*2),folder2=((folder*2)+1)))
        test=get_histograms_for_querys(    path_to_sqlite,
                                            """WHERE vector_element.hash_for_query_elo7_id<15 AND 
                                                     vector_element.hash_for_query_elo7_id IN ({folder1},{folder2})""".format(folder1=(folder*2),folder2=((folder*2)+1)),
                                            """WHERE query_elo7.hash_for_query_elo7_id<15 AND
                                                     query_elo7.hash_for_query_elo7_id IN ({folder1},{folder2})""".format(folder1=(folder*2),folder2=((folder*2)+1)),
                                            """WHERE vector_element.hash_for_query_elo7_id<15 AND
                                                     vector_element.hash_for_query_elo7_id NOT IN ({folder1},{folder2})""".format(folder1=(folder*2),folder2=((folder*2)+1)))
        return {"test":test,"train":train}




#%%
ret_from_get_histograms_for_querys_from_validation= get_histograms_for_querys_from_validation(path_to_sqlite,2)
ret_from_get_histograms_for_querys_for_train_and_test=get_histograms_for_querys_for_train_and_test(path_to_sqlite,2)
data_validation_regress=ret_from_get_histograms_for_querys_from_validation['querys_info']
data_train_test_regress={'train':ret_from_get_histograms_for_querys_for_train_and_test['train']['querys_info'],
                         'test':ret_from_get_histograms_for_querys_for_train_and_test['test']['querys_info']}

df_train=create_dataframe_for_angles_from_small_group_and_some_columns_from_table_query_elo7(data_train_test_regress['train'],'price_groups')
df_train=df_train.dropna()
columnsToUse=['angle_0','angle_1', 'angle_2', 'angle_3', 'angle_4', 'angle_5', 'angle_6','angle_7']


df_train['price']=df_train['price'].apply(np.log)
y_train=df_train['price']
df_train[columnsToUse]=df_train[columnsToUse]

df_validation=create_dataframe_for_angles_from_small_group_and_some_columns_from_table_query_elo7(data_validation_regress,'price_groups')
df_validation=df_validation.dropna()

df_validation['price']=df_validation['price'].apply(np.log)
values_from_validation_dataset_for_regression_model=df_validation['price']
df_validation[columnsToUse]=df_validation[columnsToUse]

regression_model_for_price = linear_model.LinearRegression()
regression_model_for_price.fit(df_train, y_train)
predicted_values_for_validation_dataset_for_regression_model = regression_model_for_price.predict(df_validation) 
group_vectors_fp=open( "data/models/predict_price/group_vectors.pickle", "wb" )
pickle.dump( list(ret_from_get_histograms_for_querys_from_validation['group_vectors']), group_vectors_fp)
group_vectors_fp.close()


model_predict_price_fp=open( "data/models/predict_price/mode.pickle", "wb" )
pickle.dump( regression_model_for_price, model_predict_price_fp)
model_predict_price_fp.close()


#%%[markdown]
# Por causa do bom resultado obtido foi analisado se existia 
# contaminação de dados entre os conjuntos de dados utilizados
# para treinar e validar o modelo, caso deseje repetir esta analise
# defina como *True* a variável abaixo:
#%%
checkIfExistsContaminationInDataUsedToCreateModelAndDataUsedToValidationModel=False
#%%
#code for debug purpose===========================================================================
if(checkIfExistsContaminationInDataUsedToCreateModelAndDataUsedToValidationModel):
    id_of_querys_used_to_validation_model=np.array(list(ret_from_get_histograms_for_querys_from_validation['querys_info'].keys()),dtype=np.int64)
    id_of_querys_used_to_create_vectors_models=np.array(ret_from_get_histograms_for_querys_from_validation['querys_used_for_create_histgram_of_group_of_vectors'],dtype=np.int64)
    id_of_querys_used_to_train=np.array(list(data_train_test_regress['train'].keys()),dtype=np.int64)
    id_of_querys_used_to_create_vectors_models_copy=np.array(ret_from_get_histograms_for_querys_for_train_and_test['train']['querys_used_for_create_histgram_of_group_of_vectors'],dtype=np.int64)

    for id0 in id_of_querys_used_to_validation_model:
        if(id0 in id_of_querys_used_to_create_vectors_models):
            print('Contamination for first step of checking')
            break



    for id0 in id_of_querys_used_to_create_vectors_models:
        if(not (id0 in id_of_querys_used_to_train)):
            print('Contamination for second step of checking')
            break



    for id0 in id_of_querys_used_to_create_vectors_models_copy :
        if( id0 in id_of_querys_used_to_validation_model):
            print('Contamination for third step of checking')
            break
    print("Check finished")
#end==============================================================================================




#%%[markdown]
## Avaliação do Sistema de Classificação
### Sistema de Classificação de Produtos
# A matriz de confusão foi a principal ferramenta para fazer esta avaliação. <br><br>
# A seguir a matriz de confusão do obtida no modelo de predição desenvolvido para o *Sistema de Classificação de Produtos*
#%%


#########display(confusion_mat_0)
#%%[markdown]
# Plotando um histograma para cada uma das linhas da matriz pode-se ver de forma mais direta que tipos de produtos 
# foram os mais incorretamente classificados além disso, é possível observar quais são conjuntos de classes que 
# classificador mais se confude. A seguir o plot dos histogramas:
#%% 
#########columns=confusion_mat_0.columns
#########for i in range(len(columns)):
#########    fig, ax = plt.subplots()
#########    # the size of A4 paper
#########    fig.set_size_inches(11.7, 8.27)
#########    line=confusion_mat_0.iloc[i].values
#########    df_in_dict={'valor':[],'categoria':[],'classificação correta':[]}
#########    for j in range(len(columns)):        
#########        df_in_dict['valor']=df_in_dict['valor']+([1]*int(line[j]))
#########        df_in_dict['categoria']=df_in_dict['categoria']+([columns[j][1].replace(" ","\n")]*int(line[j]))
#########        if(i!=j):
#########            df_in_dict['classificação correta']=df_in_dict['classificação correta']+(["Não"]*int(line[j]))
#########        else:
#########            df_in_dict['classificação correta']=df_in_dict['classificação correta']+(["Sim"]*int(line[j]))
#########    df_plot=pd.DataFrame.from_dict(df_in_dict)        
#########    sns.histplot(data=df_plot,x="categoria", hue="classificação correta",ax=ax)
#########    plt.show()

# %% [markdown]
# Observando-se os histogramas percebe que a categoria *Papel e Cia* é muito confundida com lembrancinhas,
# possívelmente porque muitas vendedores colocam Posters e Cartões de visita (aniversario, fim de ano e etc)
# como *Papel e Cia* e não como *Lembrancinhas*. Isso poderia ser comprovado pela analise dos histogramas dos
# produtos que são *Papel e Cia* mas que foram classificados como *Lembrancinhas*. <br><br>
# Outra categoria em que o sistema apresenta problemas para classifcar é o *Outros*, isso pode acontecer por 
# causa da diversidade dos produtos nesta categoria e por existir poucos produtos que pertencem a ela portanto,
# torna-se dificil para um modelo aprender como o atributos de entrada do modelo se relacionam com ela.


#%%[markdown]
### Sistema de termos de busca 
# Esta seção aborda os sistemas predição desenvolvido para a intenção compra de produtos de uma determinada faixa de
# preço com base naquilo que o usuário digita na consulta. Dois caminhos foram explorados para a criação desta tarefa:
# <ul>
#   <li>Utilizando árvore de decisão para predizer a intenção de compra do produto, com a predição sendo realizada sobre grupos/classes de preço. 
#       Com a entrada do modelo sendo o histograma de palavras que o usuário digita na consulta.</li>
#   <li>Utilizando regressão linear por mínimos quadrados para predizer a intenção de compra do produto, com a predição sendo realizada sobre a variável preço transformada em escala logarítmica. 
#       Com a entrada do modelo sendo o angulo do entre os histogramas médios por grupo de preço com o histograma de palavras que o usuário digita na consulta.</li>
# </ul>
# da intenção de 
#### Predição utilizando árvore de decisão para predizer a intenção de compra do produto
# A matriz de confusão foi a principal ferramenta para fazer esta avaliação. <br><br>
# A seguir a matriz de confusão obtida no modelo de predição desenvolvido:

#%%

#########conn=sqlite3.connect(path_to_sqlite)
#########df=pd.read_sql_query("""SELECT  price_group,
#########                                MIN(price),
#########                                MAX(price)
#########                        FROM query_elo7
#########                        GROUP BY price_group
#########                        ORDER BY MIN(price),MAX(price)
#########                        """,conn)
#########display(df)
#########display(confusion_mat_2)

#%%[markdown]
# Analisando a matriz de confusão é possível observar 
# que em todas pesquisas que resultaram na compra de produtos abaixo de *6,30* 
# o modelo desenvolvido presume que são pesquisas cuja a intenção 
# de compra esta entre *6,39* e *16,00*, isso mostra que ele é ruim na predição valores 
# intenção de intenção de compra abaixo de  *6,30*. <br>
# Com a análise de acuracia pode-se observar que o modelo desenvolvido
# não é muito preciso contudo, quando ele erra na predição ele tende a predizer grupos
# que são próximos do grupo original (isso pode ser observado analisando a matriz de confusão).

#### Predição utilizando regressão linear por mínimos quadrados para predizer a intenção de compra do produto
# Como este modelo trabalha com regressão então foram utilizadas as seguintes métricas para medir sua qualidade:
# <ul>
# <li> O total de diferença entre os valores reais versus o valores preditos. </li>
# <li> O media da diferença entre os valores reais versus o valores preditos. </li>
# <li> O desvio padrão da diferença entre os valores reais versus o valores preditos. </li>
# <li> O plot de um Boxplot para as diferenças entre os valores reais versus o valores preditos. </li>
# </ul>

#%%
total_difference_for_prices_predicted_and_real=sum(abs(np.exp(predicted_values_for_validation_dataset_for_regression_model)-np.exp(values_from_validation_dataset_for_regression_model)))
avg_difference_for_price_predicted_and_real=np.average(abs(np.exp(predicted_values_for_validation_dataset_for_regression_model)-np.exp(values_from_validation_dataset_for_regression_model)))
std_difference_for_price_predicted_and_real=np.std(abs(np.exp(predicted_values_for_validation_dataset_for_regression_model)-np.exp(values_from_validation_dataset_for_regression_model)))
array_of_differences_for_prices_predicted_and_real=abs(np.exp(predicted_values_for_validation_dataset_for_regression_model)-np.exp(values_from_validation_dataset_for_regression_model))

print("Diferença total de real versus o predito: "+str(total_difference_for_prices_predicted_and_real))
print("Diferença média de real versus o predito: "+str(avg_difference_for_price_predicted_and_real))
print("Desvio padrão para média de real versus o predito: "+str(avg_difference_for_price_predicted_and_real))
print("BoxPlot da diferença para real versus predito:")
sns.boxplot(x=(array_of_differences_for_prices_predicted_and_real),showfliers = False)






# %%[markdown]
# Conforme pode ser observado este modelo apresenta um otima qualidade de predição.

#%%[markdown]
# #Colaboração entre os sistemas
# Para esta etapa foi elaborado um sistema que busca onde o usuário digita algo o sistema criado em
# *Sistema de Classificação de Produtos* infere a que categoria pertence a consulta e o sistema de
# criado em *Sistema de termos de busca* acha produtos para a faixa de preço predita 
# e que foram filtrados para categoria inferida. <br> <br>
# O sistema *Sistema de termos de busca* foi feito desde o inicio utilizando apenas os termos de busca 
# das consultas, logo não precisa de modificações. Conforme análises sugerem o sistema criado em
# *Sistema de termos de busca* pode classificar produtos apenas utilizando os termos de busca digitados
# pelo usuário sem perda na acurácia.
#%%


