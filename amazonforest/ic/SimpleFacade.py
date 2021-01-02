import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, pie, draw, scatter
import os
#
from amazonforest.ic.Modeling.SimpleNB import SimpleNB
from amazonforest.ic.Modeling.SimpleRF import SimpleRF
from amazonforest.ic.Modeling.SimpleRFview import SimpleRFview
from amazonforest.ic.Modeling.SimpleRFpredict import SimpleRFpredict
from amazonforest.ic.Modeling.SimpleSVC import SimpleSVC

from amazonforest.ic.predata.SimplePreProcess import SimplePreProcess
from amazonforest.ic.Modeling.SimpleEnum import CategoryEnum

from amazonforest.ic.Modeling.SimpleMetrics import SimpleMetrics

class SimpleFacade:
    def __init__(self):
        pass
    
    def getSimpleNB(self,typeCategory):
        basedir = os.path.abspath(os.path.dirname(__file__))

        if(typeCategory == CategoryEnum.OneHot):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.onehot.train.csv"
            dfData = pd.read_csv(csvfile)
            dfData.drop(['BravaID'],axis=1, inplace=True)

            colX  = ['FATHMM_D', 'FATHMM_T', 'LRT_pred_N', 'LRT_pred_U',
                'MutaAss_H', 'MutaAss_L', 'MutaAss_M',
                'MutaAss_N', 'MutaTaster_A', 'MutaTaster_D', 'MutaTaster_N',
                'MutaTaster_P', 'PROVEAN_D', 'PROVEAN_N', 'Pph2_HDIV_B', 'Pph2_HDIV_D',
                'Pph2_HDIV_P', 'Pph2_HVAR_B', 'Pph2_HVAR_D', 'Pph2_HVAR_P', 'SIFT_D',
                'SIFT_T']

        elif (typeCategory == CategoryEnum.LabelEcoder):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
            dfData = pd.read_csv(csvfile)
            dfData.drop(['MetaSVM'],axis=1, inplace=True)
            
            colX = ['FATHMM','LRT_pred','MutaAss',
                'MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']

            dfData = dfData.dropna(subset=colX)

        elif (typeCategory == CategoryEnum.LabelEcoderScalar):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
            dfData = pd.read_csv(csvfile)
            dfData.drop(['MetaSVM'],axis=1, inplace=True)

            colX = ['FATHMM','LRT_pred','MutaAss',
                'MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']
            dfData = dfData.dropna(subset=colX)


        colTarget = 'CLNSIG'
        simplenb = SimpleNB(dfData,colX,colTarget,typeCategory)
        return simplenb


    def getSimpleRF(self,typeCategory):
        basedir = os.path.abspath(os.path.dirname(__file__))

        if(typeCategory == CategoryEnum.OneHot):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.onehot.train.csv"
            dfData = pd.read_csv(csvfile)

            dfData.drop(['BravaID'],axis=1, inplace=True)

            colX  = ['FATHMM_D', 'FATHMM_T', 'LRT_pred_N', 'LRT_pred_U',
                'MutaAss_H', 'MutaAss_L', 'MutaAss_M',
                'MutaAss_N', 'MutaTaster_A', 'MutaTaster_D', 'MutaTaster_N',
                'MutaTaster_P', 'PROVEAN_D', 'PROVEAN_N', 'Pph2_HDIV_B', 'Pph2_HDIV_D',
                'Pph2_HDIV_P', 'Pph2_HVAR_B', 'Pph2_HVAR_D', 'Pph2_HVAR_P', 'SIFT_D',
                'SIFT_T']

        elif (typeCategory == CategoryEnum.LabelEcoder):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
            dfData = pd.read_csv(csvfile)
            dfData.drop(['MetaSVM'],axis=1, inplace=True)

            colX = ['FATHMM','LRT_pred','MutaAss',
                'MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']

            dfData = dfData.dropna(subset=colX)


        elif (typeCategory == CategoryEnum.LabelEcoderScalar):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
            dfData = pd.read_csv(csvfile)
            dfData.drop(['MetaSVM'],axis=1, inplace=True)

            colX = ['FATHMM','LRT_pred','MutaAss',
                'MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']
            dfData = dfData.dropna(subset=colX)

        colTarget = 'CLNSIG'
        

        simple = SimpleRF(dfData,colX,colTarget,typeCategory)
        return simple

    def getSimpleRFview(self, typeCategory):
        simpleview = SimpleRFview()
        return simpleview
    

    def getSimpleRFpredict(self, typeCategory):
        simple = SimpleRFpredict()
        return simple

    
    def getSimpleSVC(self,typeCategory):
        basedir = os.path.abspath(os.path.dirname(__file__))

        if(typeCategory == CategoryEnum.OneHot):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.onehot.train.csv"
            dfData = pd.read_csv(csvfile)
            dfData.drop(['BravaID'],axis=1, inplace=True)
            
            colX  = ['FATHMM_D', 'FATHMM_T', 'LRT_pred_N', 'LRT_pred_U',
                'MutaAss_H', 'MutaAss_L', 'MutaAss_M',
                'MutaAss_N', 'MutaTaster_A', 'MutaTaster_D', 'MutaTaster_N',
                'MutaTaster_P', 'PROVEAN_D', 'PROVEAN_N', 'Pph2_HDIV_B', 'Pph2_HDIV_D',
                'Pph2_HDIV_P', 'Pph2_HVAR_B', 'Pph2_HVAR_D', 'Pph2_HVAR_P', 'SIFT_D',
                'SIFT_T']

        elif (typeCategory == CategoryEnum.LabelEcoder):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
            dfData = pd.read_csv(csvfile)
            dfData.drop(['MetaSVM'],axis=1, inplace=True)

            colX = ['FATHMM','LRT_pred','MutaAss',
                'MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']
            dfData = dfData.dropna(subset=colX)

        elif (typeCategory == CategoryEnum.LabelEcoderScalar):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
            dfData = pd.read_csv(csvfile)
            dfData.drop(['MetaSVM'],axis=1, inplace=True)

            colX = ['FATHMM','LRT_pred','MutaAss',
                'MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']
            dfData = dfData.dropna(subset=colX)
        
        colTarget = 'CLNSIG'

        simple = SimpleSVC(dfData,colX,colTarget,typeCategory)
        return simple

    def getSimpleSVCview(self, typeCategory):
        simpleview = SimpleSVCview()
        return simpleview

    def getSimpleSVCtest(self,typeCategory):
        basedir = os.path.abspath(os.path.dirname(__file__))

        if(typeCategory == CategoryEnum.OneHot):
            #csvfile = basedir + "/data/clinvar/03_test/pdclinvar.onehot.8.test.bp.csv"
            csvfile = basedir+ "/data/clinvar/03_test/pdclinvar.onehot.min.1.test.bp.csv"
            dfData = pd.read_csv(csvfile)
            dfData.drop(['BravaID'],axis=1, inplace=True)
            
            colX  = ['FATHMM_D', 'FATHMM_T', 'LRT_pred_N', 'LRT_pred_U',
                'MetaSVM_D', 'MetaSVM_T', 'MutaAss_H', 'MutaAss_L', 'MutaAss_M',
                'MutaAss_N', 'MutaTaster_A', 'MutaTaster_D', 'MutaTaster_N',
                'MutaTaster_P', 'PROVEAN_D', 'PROVEAN_N', 'Pph2_HDIV_B', 'Pph2_HDIV_D',
                'Pph2_HDIV_P', 'Pph2_HVAR_B', 'Pph2_HVAR_D', 'Pph2_HVAR_P', 'SIFT_D',
                'SIFT_T']

        elif (typeCategory == CategoryEnum.LabelEcoder):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
            dfData = pd.read_csv(csvfile)

            colX = ['FATHMM','LRT_pred','MetaSVM','MutaAss',
                'MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']
            dfData = dfData.dropna(subset=colX)

        elif (typeCategory == CategoryEnum.LabelEcoderScalar):
            csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
            dfData = pd.read_csv(csvfile)

            colX = ['FATHMM','LRT_pred','MetaSVM','MutaAss',
                'MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']
            dfData = dfData.dropna(subset=colX)
        
        colTarget = 'CLNSIG'

        simple = SimpleSVCtest(dfData,colX,colTarget,typeCategory)
        return simple

    def getSimpleSVCpredict(self, typeCategory):
        simplepre = SimpleSVCpredict()
        return simplepre

    def getSimpleMetrics(self):
        simple = SimpleMetrics()
        return simple

    def getPreprocess(self):
        simplePre = SimplePreProcess()
        return simplePre



if __name__ == '__main__':
        basedir = os.path.abspath(os.path.dirname(__file__))        
        csvfile = basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
        dfData = pd.read_csv(csvfile)

        colX = ['FATHMM','LRT_pred','MetaSVM','MutaAss',
            'MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']
        dfData = dfData.dropna(subset=colX)
        colTarget = 'CLNSIG'

        print(dfData.head(3))

        #simplenb = SimpleNB(dfData,colX,colTarget,typeCategory)
        #return simplenb

       # simplesvm = SimpleSVM(dfData,colX,colTarget,CategoryEnum.OneHot)
