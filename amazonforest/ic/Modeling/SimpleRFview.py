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


class SimpleRFview:


    def __init__(self):
        self.data = None
        self.dfF1Acuracy = None
        self.basedir = os.path.abspath(os.path.dirname(__file__))
        self.basedir = self.basedir.replace('/Modeling','')
    
    def dataDf(self):
        csvfile = self.basedir+ "/data/clinvar/02_training/pdclinvar.ori.train.csv"
        self.data  = pd.read_csv(csvfile)
        return self.data

    def acuF1toDf(self):
        csvfile = self.basedir+ "/data/clinvar/02_training/acuracy_label_rf.csv"        
        self.dfF1Acuracy = pd.read_csv(csvfile)        
        #print(self.dfF1Acuracy)
        self.dfF1Acuracy.drop(['F1Score (micro):'],axis=1, inplace=True)
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
