import os


def exec_cmd(cmd):
        os.system(cmd)

# propiltiouracila
if __name__ == '__main__':
   path = '/Users/helberpalheta/workhp/DesenvolvimentoDoutorado/Doutorado/easytosift/backend/biotools/snpEff'
   arquivo = path + '/easyin/100_24052020173659.vcf'

   arquivoann = path + '/easyout/100_24052020173659.ann.vcf'

   arquivoannsum = path + '/easyout/100_24052020173659.ann.sum'

   arquivopre = path + '/easyout/100_24052020173659.pre.vcf'

   
   cmd = "java -jar "+ path + "/SnpSift.jar dbnsfp -a -g GRCh37.75 -db "+ path+"/db/GRCh37/dbNSFP/dbNSFP2.9.txt.gz "
   cmd = cmd + arquivo +  " > "+ arquivopre
   exec_cmd(cmd)


   cmd = "java -jar "+ path + "/snpEff.jar GRCh37.75 "+arquivo
   cmd = cmd + " -csvStats " + arquivoannsum+  " > "+ arquivoann
   #print(cmd)
   exec_cmd(cmd)
   

   

#ok
#java -Xmx4g -jar /Users/helberpalheta/workhp/DesenvolvimentoDoutorado/Doutorado/easytosift/backend/biotools/snpEff/snpEff.jar -v GRCh37.75 /Users/helberpalheta/workhp/DesenvolvimentoDoutorado/Doutorado/easytosift/backend/biotools/snpEff/easyin/100_24052020173659.vcf -csvStats /Users/helberpalheta/workhp/DesenvolvimentoDoutorado/Doutorado/easytosift/backend/biotools/snpEff/easyout/100_24052020173659.ann.sum > /Users/helberpalheta/workhp/DesenvolvimentoDoutorado/Doutorado/easytosift/backend/biotools/snpEff/easyout/100_24052020173659.ann.vcf   
