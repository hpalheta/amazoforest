def vcfPredictSift(user_id,easy_id, vcf, path):


        arquivo = path + '/easyin/' + vcf
        
        arquivopre = path + '/easyout/' + vcf.replace('.vcf','.pre.vcf')

        #java -jar SnpSift.jar dbnsfp -a -g GRCh37.75 -db ./db/GRCh37/dbNSFP/dbNSFP2.9.txt.gz -v NativeBrazilianExome.vcf > NativeBrazilianExome.annotated.vcf
        #java -jar SnpSift.jar dbnsfp -a -g GRCh37.75 -db ./db/GRCh37/dbNSFP/dbNSFP2.9.txt.gz -v /data/1kg/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz > /data/1kg/ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.annoted.vcf

        cmd = "java -jar "+ path + "/SnpSift.jar dbnsfp -a -g GRCh37.75 -db "+ path+"/db/GRCh37/dbNSFP/dbNSFP2.9.txt.gz "

        cmd = cmd + arquivo +  " > "+ arquivopre

        

        #cmd = "./snpsift.sh "+vcf+" "+ vcf.replace('.vcf','.pre.vcf')

        #print("cmd:"+cmd)
        #os.system(cmd)

        #java -jar /Users/helberpalheta/workhp/DesenvolvimentoDoutorado/Doutorado/easytosift/backend/biotools/snpEff/SnpSift.jar'

        #"dbnsfp -a -g GRCh37.75 -db"

        import subprocess
        import io

        cmd = "java"
        sift = path+"/SnpSift.jar"
        db = path+"/db/GRCh37/dbNSFP/dbNSFP2.9.txt.gz"
        
        pyjar = subprocess.Popen([cmd,'-jar',sift ,'dbnsfp','-a','-g','GRCh37.75','-db',db,arquivo],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True)
        stdout, stderr = pyjar.communicate()

        #vcf_write = open(arquivopre, "w")
        #vcf_write.write(str(stderr))
        #vcf_write.close

        #print(stdout)
        #print(stderr) 

        vcf_reader = io.StringIO(str(stdout))
        
        #extractDal =  ExtractDal()

        #vcfPrediction = EasyVcfPrediction()
        #rs ='{"req": [{"rs": "rs2941445"},{"rs": "rs2942546"}]}'

        infoAnn = {}
        infoList = []
        for record in vcf_reader:
            #print(record)
            if(record[0:1] != '#'):
                print(record)
                #basic_vcf = vcfPrediction.getPredictionJson(user_id,easy_id,record)
                #print(basic_vcf)
                #infoList.append(basic_vcf)
                #extractDal.insertInfoVcfClinvar(basic_vcf)
        
        vcf_reader.close

        


vcfPredictSift("","", "100_teste.vcf", "/Users/helberpalheta/workhp/DesenvolvimentoDoutorado/Doutorado/easytosift/backend/biotools/snpEff")
