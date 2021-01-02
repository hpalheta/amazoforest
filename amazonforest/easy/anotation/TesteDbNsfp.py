import os
import pandas as pd
import sqlite3
import sqlite3 as db


def save_dbnsfp(df):
        conn = None
        try:
            conn =  sqlite3.connect("dbnsfp.db")
            df.to_sql('dbnsfp', con=conn, if_exists='append',index=False)
        except Exception as e: 
            conn.rollback()
        finally:
            conn.close()




arquivoVcfPath = "db1000.csv"

df = pd.read_csv(arquivoVcfPath,sep='\t')
print(df.head(10))

#easyDal = EasyDal()
save_dbnsfp(df)
