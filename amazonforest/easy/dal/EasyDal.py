import sqlite3
import sqlite3 as db
from datetime import datetime

class EasyDal:

    def __init__(self):
        #self.clienteMongo = MongoClient('mongodb://10.37.129.3:27017/bravadb')
        #self.db = self.clienteMongo['bravadb']
        pass


    def insertAnnVcf(self, ann_info):
        #id = self.db.annvcf.insert_one(ann_info)
        pass

    def saveXXXX(campo):
        pass
        # try:
        #     conn = sqlite3.connect("data/seedsmatricula.db")
        #     cur = conn.cursor()
        #     sql = "update alunoficha set "+campo + "=? where id_aluno =? "
        #     cur.execute(sql, (["S",id]))
        #     #print(sql)
        #     conn.commit()
        #     return True
        # except Exception as e:
        #     print(e)
        #     conn.rollback()
        #     return False

        # finally:
        #     # return render_template("result.html",msg = msg)
        #     conn.close()

    def save_preditc(self,df):
        try:
            conn =  sqlite3.connect("data/easytosift.db")
            df.to_sql('predict', con=conn, if_exists='append',index=False)
        except Exception as e: 
            conn.rollback()
        finally:
            conn.close()

    def save_vcf(self,df):
        try:
            print(df)
            conn =  sqlite3.connect("data/easytosift.db")
            df.to_sql('vcf', con=conn, if_exists='append',index=False)
            print("save vcf")
        except Exception as e: 
            conn.rollback()
            print("save vcf ex:"+ str(e))
        finally:
            conn.close()
    
    def save_dbnsfp(self,df):
        try:
            conn =  sqlite3.connect("data/easytosift.db")
            df.to_sql('dbnsfp', con=conn, if_exists='append',index=False)
        except Exception as e: 
            conn.rollback()
        finally:
            conn.close()
    
    def get_easy_id(self):
        try:
            conn =  sqlite3.connect("data/easytosift.db")
            conn.row_factory = sqlite3.Row
            with conn as con:
                cur = con.cursor()
                sql = "insert into tbgen (easyid) values (NULL);"
                sql +="SELECT last_insert_rowid() FROM tbgen"
                print(sql)
                #cur.execute(sql,(str(id)))
                cur.execute(sql)
                conn.commit()
                rows = cur.fetchall()
                print(rows)
                #for row in rows:
                #    print(row[0]+" "+row[1]+" "+row[2]+" "+row[3])
                #one = rows[0]
                easyid = datetime.now().strftime('%Y%m') + "_" +rows[0][0]
                return easyid
        except Exception as e: 
            #print(e)
            conn.rollback()
            rows = cur.fetchall()
        
            return rows

        finally:
            con.close()

    def getpredict_use(self,user_id ):
        try:
            conn =  sqlite3.connect("data/easytosift.db")
            conn.row_factory = sqlite3.Row
            with conn as con:
                cur = con.cursor()
                sql = "select EASYID FROM predict where user_id ="+ user_id
                cur.execute(sql)
                rows = cur.fetchall()
                #for row in rows:
                #    print(row[0]+" "+row[1]+" "+row[2]+" "+row[3])
                #one = rows[0]
                return rows
        except Exception as e: 
            #print(e)
            conn.rollback()
            rows = cur.fetchall()
        
            return rows

        finally:
            con.close()

    def dict_factory(self,cursor, row):
        d = {}
        for idx, col in enumerate(cursor.description):
            #print("idx:"+idx+" col:"+col)
            d[col[0]] = row[idx]
        return d


    def getpredict_id(self,easy_id):
        try:
            conn =  sqlite3.connect("data/easytosift.db")
            #conn.row_factory = sqlite3.Row
            conn.row_factory = self.dict_factory
            with conn as con:
                cur = con.cursor()
                sql = "select "
                #sql +=" USERID,"
                sql +=" EASYID,CHROM, POS, ID, REF, ALT, QUAL, FILTER, dbNSFP_CADD_phred CADD, dbNSFP_FATHMM_pred FATHMM"
                sql +=", dbNSFP_GERP___NR GERPNR, dbNSFP_GERP___RS GERPRS,"
                sql +=" dbNSFP_LRT_pred LRT, dbNSFP_MetaSVM_pred MetaSVM,  dbNSFP_MutationAssessor_pred MutationAssessor"
                sql +=" ,  dbNSFP_MutationTaster_pred MutationTaster,  dbNSFP_PROVEAN_pred PROVEAN, "
                sql +=" dbNSFP_Polyphen2_HDIV_pred Polyphen2_HDIV,  dbNSFP_Polyphen2_HVAR_pred Polyphen2_HVAR,  dbNSFP_SIFT_pred SIFT"
                #sql += " from predict "
                sql += " from predict where EASYID ='"+ easy_id+"'"
                #cur.execute(sql,(str(id)))
                cur.execute(sql)
                rows = cur.fetchall()
                #for row in rows:
                    #print(row)
                    #print(row[0]+" "+row[1]+" "+row[2]+" "+row[3])
                #one = rows[0]
                return rows
        except Exception as e: 
            #print(e)
            conn.rollback()
            rows = None
            return rows
        finally:
            conn.close()

    def getrequest_uid(self,userid):
        try:
            conn =  sqlite3.connect("data/easytosift.db")
            #conn.row_factory = sqlite3.Row
            conn.row_factory = self.dict_factory
            with conn as con:
                cur = con.cursor()
                sql = "select "
                sql +=" distinct EASYID,USERID,"
                sql +=" (substr(DHREGISTER,7,2)  || '/'  || substr(DHREGISTER,5,2) || '/'|| substr(DHREGISTER,1,4)"
                sql +=" || '  ' || substr(DHREGISTER,9,2) || ':' ||  substr(DHREGISTER,11,2) || ' :' ||  substr(DHREGISTER,13,2)) as DHREGISTER"
                sql += ",(select ID from vcf t2 where t1.EASYID = t2.EASYID LIMIT 1) RS"
                #sql +=" EASYID,USERID,CHROM, POS, ID, REF, ALT, QUAL, FILTER"
                sql += " from vcf t1 where USERID ='"+ userid+"'"
                sql += " order by cast(DHREGISTER AS INTEGER)  desc"
                print(sql)
                cur.execute(sql)
                rows = cur.fetchall()
                #for row in rows:
                #    print(row)
                #    print(row[0])
               


                return rows
        except Exception as e:
            #print(e)
            conn.rollback()
            rows = None
            return rows
        finally:
            conn.close()

    def getrequest_id(self,easy_id):
        try:
            conn =  sqlite3.connect("data/easytosift.db")
            #conn.row_factory = sqlite3.Row
            conn.row_factory = self.dict_factory
            with conn as con:
                cur = con.cursor()
                sql = "select "
                sql +=" distinct EASYID,USERID,"
                sql +=" (substr(DHREGISTER,7,2)  || '/'  || substr(DHREGISTER,5,2) || '/'|| substr(DHREGISTER,1,4)"
                sql +=" || '  ' || substr(DHREGISTER,9,2) || ':' ||  substr(DHREGISTER,11,2) || ' :' ||  substr(DHREGISTER,13,2)) as DHREGISTER"
                sql +=",(select ID from vcf t2 where t1.EASYID = t2.EASYID LIMIT 1) RS"
                #sql +=" EASYID,USERID,CHROM, POS, ID, REF, ALT, QUAL, FILTER"
                sql += " from vcf t1 where EASYID ='"+ easy_id+"'"
                sql += " order by cast(DHREGISTER AS INTEGER)  desc"
                cur.execute(sql)
                rows = cur.fetchall()
                return rows
        except Exception as e:
            #print(e)
            conn.rollback()
            rows = None
            return rows
        finally:
            conn.close()
