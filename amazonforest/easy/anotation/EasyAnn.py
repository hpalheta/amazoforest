import os

class EasyAnn:
    def __init__(self):
        pass
    
    def vcfAnn(self, vcf, path):
        arquivo = path + '/easyin/' + vcf
        arquivoann = path + '/easyout/' + vcf.replace('.vcf','.ann.vcf')
        arquivoannsum = path + '/easyout/' + vcf.replace('.vcf','.ann.vcf')

        cmd = "java -jar "+ path + "/snpEff.jar GRCh37.75 "+arquivo
        cmd = cmd + " -csvStats " + arquivoannsum+  " > "+ arquivoann
        #print(cmd)
        os.system(cmd)
