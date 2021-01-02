import io
import os
import pandas as  pd
import numpy as np

class SimplePreProcess:



    def __init__(self):
        self.data = None
        self.colX = None
        self.coltarget = None

    def getDataOrigin(self):
        basedir = os.path.abspath(os.path.dirname(__file__))
        basedir =  basedir.replace('/predata','')
        csvfile = basedir+ "/data/clinvar/pdclinvar.onehot.test.csv"
        dfData = pd.read_csv(csvfile)
        dfData['COUNT_PRE'] = dfData.iloc[:,1:].sum(axis = 1)
        dfData = dfData[dfData['COUNT_PRE'] > 0]
        dfData.drop(['COUNT_PRE'],axis=1, inplace=True)

        cols = dfData.columns

        self.colX  = ['FATHMM_D', 'FATHMM_T', 'LRT_pred_N', 'LRT_pred_U',
            'MetaSVM_D', 'MetaSVM_T', 'MutaAss_H', 'MutaAss_L', 'MutaAss_M',
            'MutaAss_N', 'MutaTaster_A', 'MutaTaster_D', 'MutaTaster_N',
            'MutaTaster_P', 'PROVEAN_D', 'PROVEAN_N', 'Pph2_HDIV_B', 'Pph2_HDIV_D',
            'Pph2_HDIV_P', 'Pph2_HVAR_B', 'Pph2_HVAR_D', 'Pph2_HVAR_P', 'SIFT_D',
            'SIFT_T']
        self.coltarget = 'CLNSIG'

        self.data = pd.DataFrame(dfData.head(10), columns=cols)
        return self.data
