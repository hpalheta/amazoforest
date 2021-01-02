from amazonforest.easy.anotation.EasyVcf import EasyVcf
from amazonforest.easy.anotation.EasyAnn import EasyAnn
from amazonforest.easy.anotation.EasyPredict import EasyPredict

class EasyService:
    def __init__(self):
        pass
    def getSnpVCF(self, rslist):
        #chamar EasyVCf e retornr vcf para anotacao
        easyVcf = EasyVcf()
        filevcf = easyVcf.snpToVcf(rslist)

    def annotation(self, rsdict):
        pass
    

    def run_rs(self, user_id,easy_id,rslist, path):
        #chamar EasyVCF
        easyVcf = EasyVcf()
        filevcf = easyVcf.snpsToVcf(user_id,easy_id,rslist,path)
        
        #chamar EasyAnnotation
        #easyAnn = EasyAnn()
        #easyAnn.vcfAnn(filevcf,path)

        #chamar EasyPredict
        easyPredict = EasyPredict()
        easyPredict.vcfPredictSift(user_id,easy_id,filevcf,path)

        #info predict
        #easyPredict.infoPredict(user_id,easy_id,filevcf,path)

    def run_vcf(self, user_id,easy_id,vcflist, path):
        #chamar EasyVCF
        easyVcf = EasyVcf()
        filevcf = easyVcf.listToVcf(user_id,easy_id,vcflist,path)
        print("vcf re:"+filevcf)

        #chamar EasyAnnotation
        #easyAnn = EasyAnn()
        #easyAnn.vcfAnn(filevcf,path)

        #chamar EasyPredict
        easyPredict = EasyPredict()
        easyPredict.vcfPredictSift(user_id,easy_id,filevcf,path)

        #info predict
        #easyPredict.infoPredict(user_id,easy_id,filevcf,path)

    def run_rsamazon(self, user_id,easy_id,rslist, path):
        #chamar EasyVCF
        easyVcf = EasyVcf()
        filevcf = easyVcf.snpsToVcf(user_id,easy_id,rslist,path)
        
        easyPredict = EasyPredict()
        df = easyPredict.vcfPredictSiftAmazon(user_id,easy_id,filevcf,path)

        #amazonforest
        return df 

    def run_vcfamazon(self, user_id,easy_id,vcflist, path):
        #chamar EasyVCF
        easyVcf = EasyVcf()
        filevcf = easyVcf.listToVcf(user_id,easy_id,vcflist,path)

        easyPredict = EasyPredict()
        df = easyPredict.vcfPredictSiftAmazon(user_id,easy_id,filevcf,path)

        #amazonforest
        return df 


    def runamazonforest(self, user_id,easy_id ,rslist, vcflist, path):
        if (len(rslist) ==0):
            return self.run_vcfamazon(user_id,easy_id, vcflist,path)
        else:
            return self.run_rsamazon(user_id,easy_id, rslist,path)


    def run(self, user_id,easy_id ,rslist, vcflist, path):
        if (len(rslist) ==0):
            self.run_vcf(user_id,easy_id, vcflist,path)
        else:
            self.run_rs(user_id,easy_id, rslist,path)
