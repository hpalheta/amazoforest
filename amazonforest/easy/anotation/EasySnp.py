from Bio import Entrez

import xmltodict
import io

Entrez.email = "hpalheta@gmail.com"

class EasySnp:
    def __init__(self):
        pass

    def get_snp_entrez(self,q):
        """
        Takes as input an array of snp identifiers and returns
        a parsed dictionary of their data from Entrez.
        """

        response = Entrez.efetch(db='snp', id= ','.join(q),term="ALL")
        res = ""
        for elemen in response:
            res = elemen
            #print(elemen)
            break

        return res

    def getSnpInfo(self,rs):

        snpEntrez = self.get_snp_entrez([rs])
        f = io.StringIO(str(snpEntrez))
        xml = xmltodict.parse(f.read())
        docs = xml['DocumentSummary']
        dicEntrez = {} 
        for key in ['SNP_ID','ACC','DOCSUM','CHRPOS','CHRPOS_PREV_ASSM','SNP_CLASS']:
            dicEntrez[key] = docs[key]

        seq = dicEntrez["DOCSUM"]
        i = seq.find("SEQ=[")
        j = seq.find("]")
        if(i>0):
            #chaveAux = item_info[0:i]
            dicEntrez["SEQ"] = seq[i+5:j].replace('/','>',1).replace('/',',')
            #print("seq:"+ str(dicEntrez["SEQ"]))

        #SNP_ID
        #ACC - NC_000016.10
        #DOCSUM - HGVS=NC_000016.10:g.77527584T>G,NC_000016.9:g.77561481T>G|SEQ=[T/G]
        #CHRPOS - 16:77527584
        #CHRPOS_PREV_ASSM - 16:77561481
        #SNP_CLASS - snv
        #SEQ - T>G
        
        #ch37
        #aux  = dicEntrez['CHRPOS_PREV_ASSM'].split(':')

        #chr38
        aux  = dicEntrez['CHRPOS'].split(':')

        
        chrom = aux[0]
        pos = aux[1]
        sid = "RS" + str(dicEntrez['SNP_ID'])
        aux  = dicEntrez['SEQ'].split('>')
        ref = aux[0]
        alt = aux[1]
        qual = "."
        sfilter = "PASS"
        sformat = "."
        sinfo = str(dicEntrez['ACC']) +"|"+ str(dicEntrez['SNP_CLASS']) +"|"+ str(dicEntrez['DOCSUM'])
        sinfo = "."

        #basic_vcf = { 'idVCF':idVcf,  'CHROM': chrom, 'POS' : pos, 'ID': sid, 'REF':ref,'ALT':alt,'QUAL':qual,'FILTER':sfilter,'FORMAT':sformat}
        #basic_vcf = { 'CHROM': chrom, 'POS' : pos, 'ID': sid, 'REF':ref,'ALT':alt,'QUAL':qual,'FILTER':sfilter,'FORMAT':sformat,'INFO':sinfo}
        basic_vcf = { 'CHROM': chrom, 'POS' : pos, 'ID': sid, 'REF':ref,'ALT':alt,'QUAL':qual,'FILTER':sfilter,'INFO':sinfo}
        return basic_vcf
