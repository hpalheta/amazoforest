import os
import pandas as pd
from amazonforest.easy.anotation.EasyVcfPrediction import EasyVcfPrediction
from amazonforest.easy.dal.EasyDal import EasyDal
from datetime import datetime

class EasyPredict:
    def __init__(self):
        pass
    
    def vcfPredictSift(self,user_id,easy_id, vcf, path):

        arquivo = path + '/easyin/' + vcf        
        arquivopre = path + '/easyout/' + vcf.replace('.vcf','.pre.vcf')

        #versao 01
        cmd = "java -jar "+ path + "/SnpSift.jar dbnsfp -a -g GRCh37.75 -db "+ path+"/db/GRCh37/dbNSFP/dbNSFP2.9.txt.gz "
        cmd = cmd + arquivo +  " > "+ arquivopre

        
        #versao 02 
        cmd = "./snpsift.sh "+vcf+" "+ vcf.replace('.vcf','.pre.vcf')

        #versao 03
        import subprocess
        #cmd = "/usr/lib/jvm/jre/bin/java"  no server
        cmd = "java"
        sift = path+"/SnpSift.jar"
        db = path+"/db/GRCh37/dbNSFP/dbNSFP2.9.txt.gz"
        
        pyjar = subprocess.Popen([cmd,'-jar',sift ,'dbnsfp','-a','-g','GRCh37.75','-db',db,arquivo],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True)
        stdout, stderr = pyjar.communicate()

        self.saveinfoPredict(user_id,easy_id,stdout)

        #print(stdout)
        #print(stderr)

        #vcf_write = open(arquivopre, "w")
        #vcf_write.write(str(stderr))
        #vcf_write.close

    def vcfPredictSiftAmazon(self,user_id,easy_id, vcf, path):

        arquivo = path + '/easyin/' + vcf        
        arquivopre = path + '/easyout/' + vcf.replace('.vcf','.pre.vcf')

        #versao 01
        cmd = "java -jar "+ path + "/SnpSift.jar dbnsfp -a -g GRCh37.75 -db "+ path+"/db/GRCh37/dbNSFP/dbNSFP2.9.txt.gz "
        cmd = cmd + arquivo +  " > "+ arquivopre

        
        #versao 02 
        cmd = "./snpsift.sh "+vcf+" "+ vcf.replace('.vcf','.pre.vcf')

        #versao 03
        import subprocess
        #cmd = "/usr/lib/jvm/jre/bin/java"
        cmd = "java"
        sift = path+"/SnpSift.jar"

        #versao anterior 37
        #db = path+"/db/GRCh37/dbNSFP/dbNSFP2.9.txt.gz"

        #pyjar = subprocess.Popen([cmd,'-jar',sift ,'dbnsfp','-a','-g','GRCh37.75','-db',db,arquivo],
        #                stdout=subprocess.PIPE,
        #                stderr=subprocess.PIPE,
        #                universal_newlines=True)
        #stdout, stderr = pyjar.communicate()

        db = path+"/db/GRCh38/dbNSFP/dbNSFP4.0.txt.gz"

        #java -jar SnpSift.jar dbnsfp -a -g hg38 -db ./db/GRCh38/dbNSFP/dbNSFP4.0.txt.gz -v clinvar38_20200927.vcf > clinvar38_20200927.annotated.vcf


        #java -jar SnpSift.jar dbnsfp -a -g hg38 -db ./db/GRCh38/dbNSFP/dbNSFP4.0.txt.gz -v 100_25112020091950.vcf > 100_25112020091950.annotated.vcf

        pyjar = subprocess.Popen([cmd,'-jar',sift ,'dbnsfp','-a','-g','hg38','-db',db,arquivo],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True)
        stdout, stderr = pyjar.communicate()

        df = self.saveinfoPredict(user_id,easy_id,stdout)

        return df

    def saveinfoPredict(self,user_id,easy_id, vcf):
        
        import io

        vcf_reader = io.StringIO(str(vcf))

        #extractDal =  ExtractDal()

        vcfPrediction = EasyVcfPrediction()
        #rs ='{"req": [{"rs": "rs2941445"},{"rs": "rs2942546"}]}'

        infoAnn = {}
        infoList = []
        for record in vcf_reader:
            if(record[0:1] != '#'):
                #print(record)
                basic_vcf = vcfPrediction.getPredictionJson(user_id,easy_id,record)
                #print(basic_vcf)
                infoList.append(basic_vcf)
                #extractDal.insertInfoVcfClinvar(basic_vcf)
        
        vcf_reader.close

        infoAnn = {"infoann":infoList}
        df = pd.DataFrame(infoAnn["infoann"])

        easyDal = EasyDal()
        easyDal.save_preditc(df)

        return df
        
        #print("{"+str(infoAnn["infoann"])+"}")


    def infoPredict(self,user_id,easy_id, vcf, path):
        arquivopre = path + '/easyout/' + vcf.replace('.vcf','.pre.vcf')

        vcf_reader = open(arquivopre, "r")

        #extractDal =  ExtractDal()

        vcfPrediction = EasyVcfPrediction()
        #rs ='{"req": [{"rs": "rs2941445"},{"rs": "rs2942546"}]}'

        infoAnn = {}
        infoList = []
        for record in vcf_reader:
            if(record[0:1] != '#'):
                #print(record)
                basic_vcf = vcfPrediction.getPredictionJson(user_id,easy_id,record)
                #print(basic_vcf)
                infoList.append(basic_vcf)
                #extractDal.insertInfoVcfClinvar(basic_vcf)
        
        vcf_reader.close

        infoAnn = {"infoann":infoList}
        df = pd.DataFrame(infoAnn["infoann"])

        easyDal = EasyDal()
        easyDal.save_preditc(df)

        
        #print("{"+str(infoAnn["infoann"])+"}")
