from datetime import datetime
from time import gmtime, strftime
import os
import pandas as pd
from amazonforest.easy.anotation.EasySnp import EasySnp
from amazonforest.easy.dal.EasyDal import EasyDal

class EasyVcf:
    def __init__(self):
        pass

    def snpDoVcf(self,snp):
        easySnp = EasySnp()
        vcfinfo = easySnp.getSnpInfo(snp)
        vcf =""
        for k,v in vcfinfo.items():
            if(vcf ==""):
                vcf += str(v)
            else:
                vcf += "\t" + str(v)
        #print(vcf)
        return vcf

    def dictDoVcf(self,dictvcf):
        vcf =""        
        for k,v in dictvcf.items():
            vcf = vcf + str(v) + "\t"
        return vcf


    def snpsToVcf(self,user_id,easy_id, snps, path):
        userid = 100
        arquivoVcf = str(userid) + "_" + datetime.now().strftime('%d%m%Y%H%M%S') + ".vcf"
        path = path +"/easyin"
        arquivoVcfPath = os.path.join(path, arquivoVcf)
        fileout =  open(arquivoVcfPath, 'w')
        #vcf = "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT\n"
        vcf = "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO\n"
        fileout.write(vcf)
        for snp in snps:
            vcf = self.snpDoVcf(snp)
            #print(vcf)
            fileout.write(vcf+"\n")
        fileout.flush()
        fileout.close()
        
        df = pd.read_csv(arquivoVcfPath,sep='\t')
        df["EASYID"] = easy_id
        df["USERID"] = user_id
        dhregister =  datetime.now().strftime('%Y%m%d%H%M%S')
        df["DHREGISTER"] = dhregister

        #print(df.head(10))
        
        easyDal = EasyDal()
        easyDal.save_vcf(df)
        return arquivoVcf
    
    def listToVcf(self, user_id, easy_id, vcflist, path):
        userid = 100
        arquivoVcf = str(userid) + "_" + datetime.now().strftime('%d%m%Y%H%M%S') + ".vcf"
        path = path +"/easyin"
        arquivoVcfPath = os.path.join(path, arquivoVcf)
        fileout =  open(arquivoVcfPath, 'w')
        #vcf = "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT\n"
        vcf = "#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO\n"
        fileout.write(vcf)
        for vcflinha in vcflist:
            vcf = vcflinha.replace("chr","")
            print("vcf: - "+ vcf)
            fileout.write(vcf+"\n")

        fileout.flush()
        fileout.close()
        
        df = pd.read_csv(arquivoVcfPath,sep='\t')
        df["EASYID"] = easy_id
        df["USERID"] = user_id
        dhregister =  datetime.now().strftime('%Y%m%d%H%M%S')
        df["DHREGISTER"] = dhregister

        #print(df.head(10))
        
        easyDal = EasyDal()
        easyDal.save_vcf(df)
        return arquivoVcf




#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	Patient_01_Germline	Patient_01_Somatic
#1	69091	.	A	C,G	.	PASS	AC=1	GT  1/0	2/1
