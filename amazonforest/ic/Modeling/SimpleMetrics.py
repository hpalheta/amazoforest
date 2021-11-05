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

from pathlib import Path


class SimpleMetrics:


    def __init__(self):
        self.basedir = os.path.abspath(os.path.dirname(__file__))
        self.basedir = self.basedir.replace('/Modeling','')   

    #csvfile = "/Users/helberpalheta/workhp/DesenvolvimentoDoutorado/Doutorado/Simple/easytosift/ic/data/clinvar/04_predict/pdclinvar.onehot.9.predict.svc.csv"

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
        csvfileSumma = self.basedir +  "/data/clinvar/05_summary/summary_20210208.csv"
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            return dfDataTemp
        data = [[]]
        df = pd.DataFrame(data)
        return df

    def dataSumOriPlot(self):
        img = io.BytesIO()

        csvfileSumma = self.basedir +  "/data/clinvar/05_summary/summary_20210208.csv"
        
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            #sns.set(rc={'figure.figsize':(3,3)})            
            sns_plot = sns.barplot(x=dfDataTemp.CLINVAR, y=dfDataTemp.Count, palette="deep")
            sns_plot.set_yticklabels(sns_plot.get_yticks(), size = 10)
            sns_plot.figure.savefig(img, format='png')
            sns_plot.figure.clf()
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            return 'data:image/png;base64,{}'.format(graph_url)

        return None

    def dataPreencode(self):
        csvfileSumma = self.basedir +  "/data/clinvar/05_summary/summary_20210208_pre.csv"
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            return dfDataTemp
        data = [[]]
        df = pd.DataFrame(data)
        return df

    def dataPreencodePlot(self):
        img = io.BytesIO()
        csvfileSumma = self.basedir +  "/data/clinvar/05_summary/summary_20210208_pre.csv"
        
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            #sns.set(rc={'figure.figsize':(3,3)})
            
            sns_plot = sns.barplot(x=dfDataTemp.CLINVAR, y=dfDataTemp.Count, palette="deep")
            sns_plot.set_yticklabels(sns_plot.get_yticks(), size = 10)
            sns_plot.figure.savefig(img, format='png')
            sns_plot.figure.clf()
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            return 'data:image/png;base64,{}'.format(graph_url)
        return None
    
    def dataTrain(self):
        csvfileSumma = self.basedir +  "/data/clinvar/05_summary/summary_20210208_pre.csv"
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            dfClinvaPreTrain = dfDataTemp[(dfDataTemp['CLINVAR'] =='BENIGN') | (dfDataTemp['CLINVAR'] == 'PATHOGENIC')]
            return dfClinvaPreTrain
        data = [[]]
        df = pd.DataFrame(data)
        return df

    def dataTrainPlot(self):
        img = io.BytesIO()
        csvfileSumma = self.basedir +  "/data/clinvar/05_summary/summary_20210208_pre.csv"
        
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            #sns.set(rc={'figure.figsize':(3,3)})
            #dfClinvaPreTrain = dfDataTemp[(dfDataTemp['CLINVAR'] =='BENIGN') | (dfDataTemp['CLINVAR'] == 'PATHOGENIC')]
            dfClinvaPreTrain = dfDataTemp[(dfDataTemp['CLINVAR'] =='CONFLICT') | (dfDataTemp['CLINVAR'] == 'VUS')]
            sns_plot = sns.barplot(x=dfClinvaPreTrain.CLINVAR, y=dfClinvaPreTrain.Count, palette="deep")
            sns_plot.set_yticklabels(sns_plot.get_yticks(), size = 10)
            sns_plot.figure.savefig(img, format='png')
            sns_plot.figure.clf()
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            return 'data:image/png;base64,{}'.format(graph_url)
        return None


    def dataPredict(self):
        csvfileSumma = self.basedir +  "/data/clinvar/05_summary/summary_20210208_pre.csv"
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            dfClinvaPreTrain = dfDataTemp[(dfDataTemp['CLINVAR'] =='CONFLICT') | (dfDataTemp['CLINVAR'] == 'VUS')]
            return dfClinvaPreTrain
        data = [[]]
        df = pd.DataFrame(data)
        return df

    def dataPredictPlot(self):
        img = io.BytesIO()
        csvfileSumma = self.basedir +  "/data/clinvar/05_summary/summary_20210208_pre.csv"
        
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            #sns.set(rc={'figure.figsize':(3,3)})
            dfClinvaPreTrain = dfDataTemp[(dfDataTemp['CLINVAR'] =='CONFLICT') | (dfDataTemp['CLINVAR'] == 'VUS')]
            sns_plot = sns.barplot(x=dfClinvaPreTrain.CLINVAR, y=dfClinvaPreTrain.Count, palette="deep")
            sns_plot.set_yticklabels(sns_plot.get_yticks(), size = 10)
            sns_plot.figure.savefig(img, format='png')
            sns_plot.figure.clf()
            img.seek(0)
            graph_url = base64.b64encode(img.getvalue()).decode()
            return 'data:image/png;base64,{}'.format(graph_url)
        return None


    def dataMetricInf(self,model):
        print("metricas")
        metrics = []
        csvfile = self.basedir +  "/data/clinvar/02_training/acuracy_label_"+model+".csv"        
        dfDataTemp = pd.read_csv(csvfile, sep=",")

        
        if (model == "nb"):
            metrics.append("Naive Bayes")
        elif (model == "rf"):
            metrics.append("Random Fores")
        elif (model == "SVC"):
            metrics.append("SVM")

        metrics.append(dfDataTemp["Acuracy:"][0])
        metrics.append(dfDataTemp["F1Score:"][0])
        metrics.append(dfDataTemp["MCC:"][0])

        csvfile = self.basedir +  "/data/clinvar/02_training/acuracy_max_"+model+".csv"        
        dfDataTemp = pd.read_csv(csvfile, sep=",")
        
        metrics.append(dfDataTemp["Acuracy:"][0])
        metrics.append(dfDataTemp["F1Score:"][0])
        metrics.append(dfDataTemp["MCC:"][0])

        csvfile = self.basedir +  "/data/clinvar/02_training/acuracy_onehot_"+model+".csv"        
        dfDataTemp = pd.read_csv(csvfile, sep=",")
        
        metrics.append(dfDataTemp["Acuracy:"][0])
        metrics.append(dfDataTemp["F1Score:"][0])
        metrics.append(dfDataTemp["MCC:"][0])
            
        return metrics
    
    
    def dataMetric(self):

        mnb = []
        mrf = []
        msvc = []

        mnb = self.dataMetricInf("nb")
        mrf = self.dataMetricInf("rf")
        msvc = self.dataMetricInf("SVC")
        data = [mnb,mrf,msvc]
        df = pd.DataFrame(data,columns=['Model','Label Acuracy','Label F1score','Label MCC','Max Acuracy','Max F1score','Max MCC','OneHot Acuracy','OneHot F1score','OneHot MCC'])
        #print(df)
        return df

    def dataSumTraining(self):
        csvfile = self.basedir +  "/data/clinvar/02_training/pdclinvar.onehot.train.csv"

        csvfileSumma = self.basedir +  "/data/clinvar/02_training/pdclinvar.train.summa.csv"
        
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            return dfDataTemp

        
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
        df.loc['Total']= df.sum(numeric_only=True, axis=0)
        #print(df)
        df.to_csv(csvfileSumma,sep=";",index=False)
        return df
        

    def dataSumPred(self):
        csvfile = self.basedir +  "/data/clinvar/04_predict/pdclinvar.onehot.9.predict.rf.csv"

        csvfileSumma = self.basedir +  "/data/clinvar/04_predict/pdclinvar.9.predict.rf.summa.csv"
        
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            return dfDataTemp

        
        dfDataTemp = pd.read_csv(csvfile)
        dfDataTemp['CLNSIG'] = dfDataTemp['CLNSIG'].apply(self.parserSignifLabel)
        dfDataTemp['PREDICTDESC'] = dfDataTemp['PREDICT'].apply(self.parserPredictLabel)
        cols = ['CLNSIG','PREDICT','PREDICTDESC']

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


        dfsum.to_csv(csvfileSumma,sep=";",index=False)

        return dfsum

    def dataSumCIPA(self):
        csvfile = self.basedir +  "/data/clinvar/04_predict/pdclinvar.onehot.9.predict.rf.csv"

        csvfileSumma = self.basedir +  "/data/clinvar/04_predict/pdclinvar.9.predict.rf.cipa.summa.csv"
        
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            return dfDataTemp
        
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

        df.to_csv(csvfileSumma,sep=";",index=False)
        return df
    
    def dataSumVUS(self):
        csvfile = self.basedir +  "/data/clinvar/04_predict/pdclinvar.onehot.9.predict.rf.csv" #nome onehot porem eh label

        csvfileSumma = self.basedir +  "/data/clinvar/04_predict/pdclinvar.9.predict.rf.sumvus.summa.csv"
        
        fileObj = Path(csvfileSumma)
        if (fileObj.is_file()):
            dfDataTemp = pd.read_csv(csvfileSumma, sep=";")
            return dfDataTemp
        
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

        df.to_csv(csvfileSumma,sep=";",index=False)
        return df
    
    def dataDf(self):
        csvfile = self.basedir+ "/data/clinvar/02_training/pdclinvar.onehot.train.csv"
        self.data  = pd.read_csv(csvfile)
        return self.data

    def acuF1toDf(self):
        csvfile = self.basedir+ "/data/clinvar/02_training/acuracy_onehot_rf.csv"
        self.dfF1Acuracy = pd.read_csv(csvfile)
        print(self.dfF1Acuracy)
        return self.dfF1Acuracy

    def plotRoc(self, tipo):
        filename = self.basedir + "/data/clinvar/02_training/plot_svc_onehot_roc.png"
        image = Image.open(filename)
        img = io.BytesIO()
        image.save(img,"png")
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        image.close()
        return 'data:image/png;base64,{}'.format(graph_url)

    def plotCMToImg(self,tipo):
        filename = self.basedir + "/data/clinvar/02_training/plot_svc_onehot_cm.png"
        image = Image.open(filename)
        img = io.BytesIO()
        image.save(img,"png")
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        image.close()
        return 'data:image/png;base64,{}'.format(graph_url)
