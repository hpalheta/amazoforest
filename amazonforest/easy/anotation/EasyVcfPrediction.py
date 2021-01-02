
import collections

class EasyVcfPrediction:

    def __init__(self):
        self.easy_id = None
        self.user_id = None
        self.chrom= None
        self.pos  = None
        self.sid  = None
        self.ref  = None
        self.alt  = None
        self.qual = None
        self.sfilter = None
        self.dbNSFP_CADD_phred = None
        self.dbNSFP_FATHMM_pred = None
        self.dbNSFP_GERP___NR = None
        self.dbNSFP_GERP___RS = None
        self.dbNSFP_LRT_pred = None
        self.dbNSFP_MetaSVM_pred = None
        self.dbNSFP_MutationAssessor_pred = None
        self.dbNSFP_MutationTaster_pred = None
        self.dbNSFP_PROVEAN_pred = None
        self.dbNSFP_Polyphen2_HDIV_pred = None
        self.dbNSFP_Polyphen2_HVAR_pred = None
        self.dbNSFP_SIFT_pred = None


    def clear(self):
        self.easy_id = None
        self.user_id = None
        self.chrom= None
        self.pos  = None
        self.sid  = None
        self.ref  = None
        self.alt  = None
        self.qual = None
        self.sfilter = None
        self.dbNSFP_CADD_phred = None
        self.dbNSFP_FATHMM_pred = None
        self.dbNSFP_GERP___NR = None
        self.dbNSFP_GERP___RS = None
        self.dbNSFP_LRT_pred = None
        self.dbNSFP_MetaSVM_pred = None
        self.dbNSFP_MutationAssessor_pred = None
        self.dbNSFP_MutationTaster_pred = None
        self.dbNSFP_PROVEAN_pred = None
        self.dbNSFP_Polyphen2_HDIV_pred = None
        self.dbNSFP_Polyphen2_HVAR_pred = None
        self.dbNSFP_SIFT_pred = None

    def isValid(self):
        if(self.dbNSFP_CADD_phred == None):
            return False
        if(self.dbNSFP_FATHMM_pred == None):
            return False
        if(self.dbNSFP_GERP___NR == None):
            return False
        if(self.dbNSFP_GERP___RS == None):
            return False
        if(self.dbNSFP_LRT_pred == None):
            return False
        if(self.dbNSFP_MetaSVM_pred == None):
            return False
        if(self.dbNSFP_MutationAssessor_pred == None):
            return False
        if(self.dbNSFP_MutationTaster_pred == None):
            return False
        if(self.dbNSFP_PROVEAN_pred == None):
            return False
        if(self.dbNSFP_Polyphen2_HDIV_pred == None):
            return False
        if(self.dbNSFP_Polyphen2_HVAR_pred == None):
            return False
        if(self.dbNSFP_SIFT_pred == None):
            return False
        #sem erro return true
        return True

    def getPredictionJson(self,user_id,easy_id,rowvcf):
        self.clear()
        rowvcf = rowvcf.rstrip('\n')
        #print(rowvcf)
        #"CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
        split_linha = rowvcf.split('\t')
        #campos basicos
        self.userid = user_id
        self.easyid = easy_id
        self.chrom= split_linha[0]
        self.pos  = split_linha[1]
        self.sid  = split_linha[2]
        self.ref  = split_linha[3]
        self.alt  = split_linha[4]
        self.qual = split_linha[5]
        self.sfilter = split_linha[6]


        info_list = split_linha[7].split(';')

        for item_info in info_list:
            #print('Info Item:' + item_info)
            i = item_info.find('=')
            if(i>0):
                    chaveAux = item_info[0:i]
                    if (chaveAux == "dbNSFP_CADD_phred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_CADD_phred =  None
                        else:
                            self.dbNSFP_CADD_phred = item_info[i+1:]

                    if (chaveAux == "dbNSFP_FATHMM_pred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_FATHMM_pred =  None
                        else:
                            self.dbNSFP_FATHMM_pred = item_info[i+1:]

                    if (chaveAux == "dbNSFP_GERP___NR"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_GERP___NR =  None
                        else:
                            self.dbNSFP_GERP___NR = item_info[i+1:]
                    if (chaveAux == "dbNSFP_GERP___RS"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_GERP___RS =  None
                        else:
                            self.dbNSFP_GERP___RS = item_info[i+1:]
                    if (chaveAux == "dbNSFP_LRT_pred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_LRT_pred =  None
                        else:
                            self.dbNSFP_LRT_pred = item_info[i+1:]
                    if (chaveAux == "dbNSFP_MetaSVM_pred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_MetaSVM_pred =  None
                        else:
                            self.dbNSFP_MetaSVM_pred = item_info[i+1:]
                    if (chaveAux == "dbNSFP_MutationAssessor_pred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_MutationAssessor_pred =  None
                        else:
                            self.dbNSFP_MutationAssessor_pred = item_info[i+1:]
                    if (chaveAux == "dbNSFP_MutationTaster_pred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_MutationTaster_pred =  None
                        else:
                            self.dbNSFP_MutationTaster_pred = item_info[i+1:]
                    if (chaveAux == "dbNSFP_PROVEAN_pred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_PROVEAN_pred =  None
                        else:
                            self.dbNSFP_PROVEAN_pred = item_info[i+1:]
                    if (chaveAux == "dbNSFP_Polyphen2_HDIV_pred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_Polyphen2_HDIV_pred =  None
                        else:
                            self.dbNSFP_Polyphen2_HDIV_pred = item_info[i+1:]
                    if (chaveAux == "dbNSFP_Polyphen2_HVAR_pred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_Polyphen2_HVAR_pred =  None
                        else:
                            self.dbNSFP_Polyphen2_HVAR_pred = item_info[i+1:]
                    if (chaveAux == "dbNSFP_SIFT_pred"):
                        if(item_info[i+1:]==''):
                            self.dbNSFP_SIFT_pred =  None
                        else:
                            self.dbNSFP_SIFT_pred = item_info[i+1:]

        #campos iniciais
        pred_vcf = None
        #if(self.isValid()):
        pred_vcf = { "EASYID":self.easyid, "USERID":self.userid,  "CHROM": self.chrom, "POS" : self.pos,"ID": self.sid, "REF":self.ref,"ALT":self.alt,"QUAL":self.qual,"FILTER":self.sfilter}
        pred_vcf["dbNSFP_CADD_phred"] = self.dbNSFP_CADD_phred
        pred_vcf["dbNSFP_FATHMM_pred"] = self.dbNSFP_FATHMM_pred
        pred_vcf["dbNSFP_GERP___NR"] = self.dbNSFP_GERP___NR
        pred_vcf["dbNSFP_GERP___RS"] = self.dbNSFP_GERP___RS
        pred_vcf["dbNSFP_LRT_pred"] = self.dbNSFP_LRT_pred
        pred_vcf["dbNSFP_MetaSVM_pred"] = self.dbNSFP_MetaSVM_pred
        pred_vcf["dbNSFP_MutationAssessor_pred"] = self.dbNSFP_MutationAssessor_pred
        pred_vcf["dbNSFP_MutationTaster_pred"] = self.dbNSFP_MutationTaster_pred
        pred_vcf["dbNSFP_PROVEAN_pred"] = self.dbNSFP_PROVEAN_pred
        pred_vcf["dbNSFP_Polyphen2_HDIV_pred"] = self.dbNSFP_Polyphen2_HDIV_pred
        pred_vcf["dbNSFP_Polyphen2_HVAR_pred"] = self.dbNSFP_Polyphen2_HVAR_pred
        pred_vcf["dbNSFP_SIFT_pred"] = self.dbNSFP_SIFT_pred

        return pred_vcf

    def getAjustedPredictionJson(self,rowvcf):
        self.clear()
        rowvcf = rowvcf.rstrip('\n')
        #print(rowvcf)
        #"CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER",
        split_linha = rowvcf.split('\t')
        #campos basicos
        self.chrom= split_linha[0]
        self.pos  = split_linha[1]
        self.sid  = split_linha[2]
        self.ref  = split_linha[3]
        self.alt  = split_linha[4]
        self.qual = split_linha[5]
        self.sfilter = split_linha[6]
        self.bravaid = self.chrom+':'+self.pos

        info_list = split_linha[7].split(';')

        for item_info in info_list:
            #print('Info Item:' + item_info)
            i = item_info.find('=')
            if(i>0):
                    chaveAux = item_info[0:i]
                    if (chaveAux == "dbNSFP_CADD_phred"):
                        c = collections.Counter(item_info[i+1:])
                        if(c.most_common(1)[0][0]==''):
                            self.dbNSFP_CADD_phred =  None
                        else:
                            self.dbNSFP_CADD_phred = float(c.most_common(1)[0][0])

                        #stemp = item_info[i+1:]
                        #j = stemp.find(',')
                        #if(j>0):
                        #    self.dbNSFP_CADD_phred = float(stemp[0:j])
                        #else:
                        #    self.dbNSFP_CADD_phred = float(stemp)

                    if (chaveAux == "dbNSFP_FATHMM_pred"):
                        c = collections.Counter(item_info[i+1:])
                        if(c.most_common(1)[0][0]==''):
                            self.dbNSFP_FATHMM_pred =  None
                        else:
                            self.dbNSFP_FATHMM_pred = c.most_common(1)[0][0]

                    if (chaveAux == "dbNSFP_GERP___NR"):
                        c = collections.Counter(item_info[i+1:])
                        print('c-'+ str(c))
                        if(c.most_common(1)[0][0]=='.'):
                            self.dbNSFP_GERP___NR =  None
                        elif(c.most_common(1)[0][0]==''):
                            self.dbNSFP_GERP___NR =  None
                        elif(c.most_common(1)[0][0]=='-'):
                            self.dbNSFP_GERP___NR =  None
                        else:
                            self.dbNSFP_GERP___NR = float(c.most_common(1)[0][0])
                    if (chaveAux == "dbNSFP_GERP___RS"):
                        c = collections.Counter(item_info[i+1:])
                        if(c.most_common(1)[0][0]=='.'):
                            self.dbNSFP_GERP___RS =  None
                        elif(c.most_common(1)[0][0]==''):
                            self.dbNSFP_GERP___RS =  None
                        elif(c.most_common(1)[0][0]=='-'):
                            self.dbNSFP_GERP___RS =  None
                        else:
                            self.dbNSFP_GERP___RS = float(c.most_common(1)[0][0])
                    if (chaveAux == "dbNSFP_LRT_pred"):
                        c = collections.Counter(item_info[i+1:])
                        if(c.most_common(1)[0][0]==''):
                            self.dbNSFP_FATHMM_pred =  None
                        else:
                            self.dbNSFP_FATHMM_pred = c.most_common(1)[0][0]
                    if (chaveAux == "dbNSFP_MetaSVM_pred"):
                        self.dbNSFP_MetaSVM_pred = (item_info[i+1:])
                    if (chaveAux == "dbNSFP_MutationAssessor_pred"):
                        self.dbNSFP_MutationAssessor_pred = (item_info[i+1:])
                    if (chaveAux == "dbNSFP_MutationTaster_pred"):
                        self.dbNSFP_MutationTaster_pred = (item_info[i+1:])
                    if (chaveAux == "dbNSFP_PROVEAN_pred"):
                        c = collections.Counter(item_info[i+1:])
                        if(c.most_common(1)[0][0]==''):
                            self.dbNSFP_PROVEAN_pred =  None
                        else:
                            self.dbNSFP_PROVEAN_pred = c.most_common(1)[0][0]
                    if (chaveAux == "dbNSFP_Polyphen2_HDIV_pred"):
                        self.dbNSFP_Polyphen2_HDIV_pred = (item_info[i+1:])
                    if (chaveAux == "dbNSFP_Polyphen2_HVAR_pred"):
                        self.dbNSFP_Polyphen2_HVAR_pred = (item_info[i+1:])
                    if (chaveAux == "dbNSFP_SIFT_pred"):
                        c = collections.Counter(item_info[i+1:])
                        if(c.most_common(1)[0][0]==''):
                            self.dbNSFP_SIFT_pred =  None
                        else:
                            self.dbNSFP_SIFT_pred = c.most_common(1)[0][0]

           
        #campos iniciais
        pred_vcf = None
        if(self.isValid()):
            pred_vcf = { 'idVCF':str(0),'BravaID':self.easyid,  'CHROM': self.chrom, 'POS' : self.pos,'ID': self.sid, 'REF':self.ref,'ALT':self.alt,'QUAL':self.qual,'FILTER':self.sfilter}
            pred_vcf["dbNSFP_CADD_phred"] = self.dbNSFP_CADD_phred
            pred_vcf["dbNSFP_FATHMM_pred"] = self.dbNSFP_FATHMM_pred
            pred_vcf["dbNSFP_GERP___NR"] = self.dbNSFP_GERP___NR
            pred_vcf["dbNSFP_GERP___RS"] = self.dbNSFP_GERP___RS
            pred_vcf["dbNSFP_LRT_pred"] = self.dbNSFP_LRT_pred
            pred_vcf["dbNSFP_MetaSVM_pred"] = self.dbNSFP_MetaSVM_pred
            pred_vcf["dbNSFP_MutationAssessor_pred"] = self.dbNSFP_MutationAssessor_pred
            pred_vcf["dbNSFP_MutationTaster_pred"] = self.dbNSFP_MutationTaster_pred
            pred_vcf["dbNSFP_PROVEAN_pred"] = self.dbNSFP_PROVEAN_pred
            pred_vcf["dbNSFP_Polyphen2_HDIV_pred"] = self.dbNSFP_Polyphen2_HDIV_pred
            pred_vcf["dbNSFP_Polyphen2_HVAR_pred"] = self.dbNSFP_Polyphen2_HVAR_pred
            pred_vcf["dbNSFP_SIFT_pred"] = self.dbNSFP_SIFT_pred
            
        return pred_vcf


##Que tipo de processamento estao Utilizando
