

#%%[markdown]
#**Carregar base de dados para o SQLite**
#Definir localização da base de dados 
#%%
path_to_database='data/raw/elo7_recruitment_dataset.csv'
#%%[markdown]
#Definir localização onde SQLite vai ser guardado
#%%
path_to_sqlite='data/interim/database.sqlite3'


#%%[markdown]
#Passar os dados 
#%%
import src.data.load_data_to_sqlite_dataset as load_dataset
import pandas as pd
import sqlite3
import nltk
from nltk import word_tokenize
nltk.download('punkt')

def pass_csv_database_to_sqlite3(path_to_sqlite3,path_raw_data_in_csv):

    conn = sqlite3.connect(path_to_sqlite3)
    df=pd.read_csv(path_raw_data_in_csv)
    df.to_sql("query_elo7", conn, if_exists="replace",index_label='querys_elo7_id')
    conn.close()

pass_csv_database_to_sqlite3(path_to_sqlite,path_to_database)

load_dataset.create_sqlite_schema()

#%%[markdown]
#Associar individualmente cada palavras digitadas nas querys para futuras análises estatísticas
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



"""
SELECT word_typed_in_query___word_typed_in_query.word_typed_in_query_id,
       COUNT(querys_elo7_id),
       word_typed_in_query.word
  FROM word_typed_in_query___word_typed_in_query
  INNER JOIN word_typed_in_query ON word_typed_in_query.word_typed_in_query_id=word_typed_in_query___word_typed_in_query.word_typed_in_query_id
  WHERE  word_typed_in_query.word NOT IN ("da",'de','para','dos','12','13','14','15','16','17','18','20','o','a','em','na')
  GROUP BY word_typed_in_query___word_typed_in_query.word_typed_in_query_id
"""

"""
WITH  TEMP1 AS (
SELECT word_typed_in_query___word_typed_in_query.word_typed_in_query_id AS id_word,
       COUNT(querys_elo7_id) AS c,
       word_typed_in_query.word
  FROM word_typed_in_query___word_typed_in_query
  INNER JOIN word_typed_in_query ON word_typed_in_query.word_typed_in_query_id=word_typed_in_query___word_typed_in_query.word_typed_in_query_id
  WHERE  word_typed_in_query.word NOT IN ("da",'de','para','dos','12','13','14','15','16','17','18','20','o','a','em','na')
  
  GROUP BY word_typed_in_query___word_typed_in_query.word_typed_in_query_id
),
TEMP2 AS (
    SELECT *
    FROM TEMP1
    WHERE c>26
)
SELECT DISTINCT word_typed_in_query___word_typed_in_query.querys_elo7_id
FROM word_typed_in_query___word_typed_in_query
INNER JOIN TEMP2 ON TEMP2.id_word=word_typed_in_query___word_typed_in_query.word_typed_in_query_id
"""
create_schema_for_tables_that_associate_words_in_querys_with_querys_typed(path_to_sqlite)
populate_table__word_typed_in_query(path_to_sqlite)
create_schema_for_table___word_typed_in_query___word_typed_in_query(path_to_sqlite)
word_typed_in_query___word_typed_in_query(path_to_sqlite)

