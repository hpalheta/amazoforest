import io
import os
import base64
import pandas as  pd
import numpy as np
from scipy import interp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure, plot, bar, pie, draw, scatter
from amazonforest.ic.Modeling.SimpleEnum import CategoryEnum

from PIL import Image, ExifTags

import sklearn

# Tipos de NB
from sklearn.naive_bayes import MultinomialNB, ComplementNB, BernoulliNB, GaussianNB

# Metricas utilizadas
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, confusion_matrix

# pre processamento - normalizacao - padronizacao
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Validacao cruzada
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc
#
from sklearn.utils.multiclass import unique_labels
#
from sklearn.model_selection import learning_curve, GridSearchCV


class SimpleRFpredict:


    def __init__(self):
        self.basedir = os.path.abspath(os.path.dirname(__file__))
        self.basedir = self.basedir.replace('/Modeling','')   

   
    def parserSignifLabel(self,valor):
        valor = str(valor)
        if (valor == '1'):
            return 'BENIG'
        elif ( valor == '2'):
            return 'PATHO'
        elif ( valor == '3'):
            return 'CIPAT'
        else:
            return 'VUS'

    def parserPredictLabel(self,valor):
        valor = str(valor)
        if(valor == '0.0'):
            return 'BENIG'
        else :
            return 'PATHO'


    def dataSumOri(self):
        csvfile = self.basedir +  "/data/clinvar/00_original/pdclinvar38_20200927.pre.csv"
        
        dfDataTemp = pd.read_csv(csvfile, sep=";")

        cols = ['BravaID','CLNSIG']
        dfDataOri = dfDataTemp[cols]

        ser1 = dfDataOri['CLNSIG'].value_counts()
        ser1per = dfDataOri['CLNSIG'].value_counts(normalize=True)


        clnsig = []
        count = []
        percent = []

        for i in range(0,len(ser1.index),1):
            clnsig.append(ser1.index[i])
            count.append(ser1[i])
            percent.append(ser1per[i])
        
        d = {'CLNSIG': pd.Series(clnsig),'Count':pd.Series(count),'%': pd.Series(percent)}
        df = pd.DataFrame(d)

        df.loc['Total']= df.sum(numeric_only=True, axis=0)
        return df

    def dataSumTraining(self):
        csvfile = self.basedir +  "/data/clinvar/02_training/pdclinvar.ori.train.csv"
        
        dfDataTemp = pd.read_csv(csvfile)
        dfDataTemp['CLNSIG'] = dfDataTemp['CLNSIG'].apply(self.parserSignifLabel)
        cols = ['CLNSIG']        
        dfDataOri = dfDataTemp[cols]
        
        ser1 = dfDataOri['CLNSIG'].value_counts()
        ser1per = dfDataOri['CLNSIG'].value_counts(normalize=True)

        clnsig = []
        count = []
        percent = []

        for i in range(0,len(ser1.index),1):
            clnsig.append(ser1.index[i])
            count.append(ser1[i])
            percent.append(ser1per[i])
        
        d = {'CLNSIG': pd.Series(clnsig),'Count':pd.Series(count),'%': pd.Series(percent)}
        df = pd.DataFrame(d)
        #print(df)
        df.loc['Total']= df.sum(numeric_only=True, axis=0)
        return df

    def dataSumPred(self):
        csvfile = self.basedir +  "/data/clinvar/04_predict/pdclinvar.predict.rf.csv"
        
        dfDataTemp = pd.read_csv(csvfile)
        dfDataTemp['CLNSIG'] = dfDataTemp['CLNSIG'].apply(self.parserSignifLabel)
        dfDataTemp['PREDICTDESC'] = dfDataTemp['PREDICT'].apply(self.parserPredictLabel)
        cols = ['BravaID','CLNSIG','PREDICT','PREDICTDESC']
        dfDatapredict = dfDataTemp[cols]

        df1= dfDatapredict[['CLNSIG','PREDICTDESC']].value_counts()
        df1per= dfDatapredict[['CLNSIG','PREDICTDESC']].value_counts(normalize=True)

        clnsig = []
        predict = []
        count = []
        percent = []

        for i in range(0,len(df1.index),1):
            id = df1.index[i]
            clnsig.append(id[0])
            predict.append(id[1])
            count.append(df1[i])
            percent.append(df1per[i])

        d = {'SIGN': pd.Series(clnsig),'PREDICT': pd.Series(predict),'COUNT': pd.Series(count),'%': pd.Series(percent)}
        dfsum = pd.DataFrame(d)
        dfsum.loc['Total']= dfsum.sum(numeric_only=True, axis=0)

        return dfsum

    def dataSumCIPA(self):
        csvfile = self.basedir +  "/data/clinvar/04_predict/pdclinvar.predict.rf.csv"
        
        dfDataTemp = pd.read_csv(csvfile)
        dfDataTemp['CLNSIG'] = dfDataTemp['CLNSIG'].apply(self.parserSignifLabel)
        dfDataTemp = dfDataTemp[(dfDataTemp['CLNSIG']=='CIPAT')]
        dfDataTemp['PREDICTDESC'] = dfDataTemp['PREDICT'].apply(self.parserPredictLabel)
        cols = ['PREDICTDESC']        
        dfDataOri = dfDataTemp[cols]
        
        ser1 = dfDataOri['PREDICTDESC'].value_counts()
        ser1per = dfDataOri['PREDICTDESC'].value_counts(normalize=True)

        clnsig = []
        count = []
        percent = []

        for i in range(0,len(ser1.index),1):
            clnsig.append(ser1.index[i])
            count.append(ser1[i])
            percent.append(ser1per[i])
        
        d = {'PREDICT': pd.Series(clnsig),'Count':pd.Series(count),'%': pd.Series(percent)}
        df = pd.DataFrame(d)
        df.loc['Total']= df.sum(numeric_only=True, axis=0)
        return df
    
    def dataSumVUS(self):
        csvfile = self.basedir +  "/data/clinvar/04_predict/pdclinvar.predict.rf.csv"
        
        dfDataTemp = pd.read_csv(csvfile)
        dfDataTemp['CLNSIG'] = dfDataTemp['CLNSIG'].apply(self.parserSignifLabel)
        dfDataTemp = dfDataTemp[(dfDataTemp['CLNSIG']=='VUS')]
        dfDataTemp['PREDICTDESC'] = dfDataTemp['PREDICT'].apply(self.parserPredictLabel)
        cols = ['PREDICTDESC']        
        dfDataOri = dfDataTemp[cols]
        
        
        ser1 = dfDataOri['PREDICTDESC'].value_counts()
        ser1per = dfDataOri['PREDICTDESC'].value_counts(normalize=True)

        clnsig = []
        count = []
        percent = []

        for i in range(0,len(ser1.index),1):
            clnsig.append(ser1.index[i])
            count.append(ser1[i])
            percent.append(ser1per[i])
        
        d = {'PREDICT': pd.Series(clnsig),'Count':pd.Series(count),'%': pd.Series(percent)}
        df = pd.DataFrame(d)

        df.loc['Total']= df.sum(numeric_only=True, axis=0)
        return df
    
    def dataDf(self):
        csvfile = self.basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
        self.data  = pd.read_csv(csvfile)
        return self.data

    def acuF1toDf(self):
        csvfile = self.basedir+ "/data/clinvar/02_training/acuracy_label_rf.csv"
        self.dfF1Acuracy = pd.read_csv(csvfile)
        print(self.dfF1Acuracy)
        return self.dfF1Acuracy

    def plotRoc(self, tipo):
        filename = self.basedir + "/data/clinvar/02_training/plot_rf_label_roc.png"
        image = Image.open(filename)
        img = io.BytesIO()
        image.save(img,"png")
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        image.close()
        return 'data:image/png;base64,{}'.format(graph_url)

    def plotCMToImg(self,tipo):
        filename = self.basedir + "/data/clinvar/02_training/plot_rf_label_cm.png"
        image = Image.open(filename)
        img = io.BytesIO()
        image.save(img,"png")
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        image.close()
        return 'data:image/png;base64,{}'.format(graph_url)

    def dataPredictCutOff(self, chrom):
        csvfile = self.basedir+ "/data/clinvar/04_predict/pdclinvar.predict.cutoff90.rf.csv"
        data  = pd.read_csv(csvfile)
        dfDataTemp = data[(data['CHROM']==chrom)]

        dfDataTemp["PROBA"] =  dfDataTemp["1"]



        cols = ['PROBA','CHROM', 'POS', 'ID', 'REF', 'ALT','CLNSIGALL', 'FATHMM', 'LRT_pred', 'MutaAss', 'MutaTaster','PROVEAN', 'Pph2_HDIV', 'Pph2_HVAR', 'SIFT', 'PREDICT']
        data = dfDataTemp[cols]

        return data

    def NewPredict_RF(self,fathmm,lrt,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift):
        import pickle
        
        basedir = os.path.abspath(os.path.dirname(__file__))
        basedir = basedir.replace('Modeling','')

        print(basedir)
        
        d = {'FATHMM': [fathmm] ,'LRT_pred':[lrt],'MutaAss': [mutass],'MutaTaster':[muttaster],'PROVEAN':[provean],'Pph2_HDIV':[pph2_hdiv],
            'Pph2_HVAR':[pph2_hvar],'SIFT':[sift]}
        
        dfDataHomo = pd.DataFrame.from_dict(d)
        colX = ['FATHMM','LRT_pred','MutaAss','MutaTaster', 'PROVEAN', 'Pph2_HDIV','Pph2_HVAR',  'SIFT']
        
        dfData = pd.DataFrame(dfDataHomo[colX])
        
        dfData = dfData.dropna(subset=colX)

        labelencoder_col = LabelEncoder()
       
        labelencoder_col.fit(['D', 'T'])
        dfData['FATHMM'] = labelencoder_col.transform(dfData['FATHMM'])
        labelencoder_col.fit(['N', 'U', 'Z']) 
        dfData['LRT_pred'] = labelencoder_col.transform(dfData['LRT_pred'])
        labelencoder_col.fit(['H', 'L', 'M', 'N'])
        dfData['MutaAss'] = labelencoder_col.transform(dfData['MutaAss'])
        labelencoder_col.fit(['A', 'D', 'N', 'P'])
        dfData['MutaTaster'] = labelencoder_col.transform(dfData['MutaTaster'])
        labelencoder_col.fit(['D', 'N'])
        dfData['PROVEAN'] = labelencoder_col.transform(dfData['PROVEAN'])
        labelencoder_col.fit(['B', 'D', 'P'])
        dfData['Pph2_HDIV'] = labelencoder_col.transform(dfData['Pph2_HDIV'])
        labelencoder_col.fit(['B', 'D', 'P'])
        dfData['Pph2_HVAR'] = labelencoder_col.transform(dfData['Pph2_HVAR'])
        labelencoder_col.fit(['D', 'T'])
        dfData['SIFT'] = labelencoder_col.transform(dfData['SIFT'])

        x = dfData[colX].to_numpy()

        filemodel = basedir+ "/data/clinvar/02_training/predict_rf_lbl.sav"
        loaded_model = pickle.load(open(filemodel, 'rb'))
        result = loaded_model.predict(x) 
        result2 = loaded_model.predict_proba(x)

        dfDataProba = pd.DataFrame(result2)
        
        dfDataHomo['PREDICT'] = result
        dfDataHomo['0'] = dfDataProba[0]
        dfDataHomo['1'] = dfDataProba[1]

        print(dfDataHomo)

        return dfDataHomo
