import io
import base64
import pandas as  pd
import numpy as np
from scipy import interp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, pie, draw, scatter

#Import SVM

from sklearn.svm import SVC

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

from amazonforest.ic.Modeling.SimpleEnum import CategoryEnum

class SimpleSVC:
    def __init__(self,data,colX,colTarget,typeCategory):
        self.data = data
        self.colX = colX
        self.coltarget = colTarget
        self.typeCategory = typeCategory
        self.tprs = []
        self.aucs = []
        self.mean_fpr =None
        self.ytests = []
        self.ypreds = []
        self.dfF1Acuracy = None
        self.fpr_tpr_list = []


    def plot_confusion_matrix(self,y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        #classes = classes[unique_labels(y_true, y_pred)]



        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        #print(cm)

        
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
                
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
            # ... and label them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='Benign',
            xlabel='Pathogenic')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    def plotCM(self,tipo):
        confusion_matrix(self.ytests, self.ypreds).ravel()
        np.set_printoptions(precision=2)
        #classes = np.asarray(self.coltarget)
        classes = [0.0,0.1]


        # Plot normalized confusion matrix
        self.plot_confusion_matrix(self.ytests, self.ypreds, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

        img = 'easytosift/ic/data/clinvar/02_training/plot_svc_'+tipo+'_cm.sav'
        plt.savefig(img, format='png')
        plt.close()
   
    def plotCMToImg(self):
        img = io.BytesIO()
        plt.rcParams['figure.figsize'] = (11,7)

        confusion_matrix(self.ytests, self.ypreds).ravel()
        np.set_printoptions(precision=2)

        classes = [0.0,0.1]

        # Plot normalized confusion matrix
        self.plot_confusion_matrix(self.ytests, self.ypreds, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(graph_url)

    def AcuF1toDf(self):
        return self.dfF1Acuracy

    def plotRoc(self, tipo):
        plt.rcParams['figure.figsize'] = (11,7)
        i= 0
        for roc_auc in self.aucs:            
            
            fpr = self.fpr_tpr_list[i][0]
            tpr = self.fpr_tpr_list[i][1]
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            i+=1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
          label='Chance', alpha=.8)

        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        plt.plot(self.mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Prediction')
        plt.legend(loc="lower right")

        img = 'easytosift/ic/data/clinvar/02_training/plot_svc_'+tipo+'_roc.sav'
        plt.savefig(img, format='png')
        plt.close()

    def plotRocToImg(self):
        img = io.BytesIO()
        plt.rcParams['figure.figsize'] = (11,7)
        i= 0
        for roc_auc in self.aucs:
            fpr = self.fpr_tpr_list[i][0]
            tpr = self.fpr_tpr_list[i][1]
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
            i+=1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        plt.plot(self.mean_fpr, mean_tpr, color='b',
                label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                lw=2, alpha=.8)

        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Prediction')
        plt.legend(loc="lower right")

        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return 'data:image/png;base64,{}'.format(graph_url)


    def plotBar(self):
        plt.bar(self.ytests, self.ypreds)
        plt.xlabel("True Values")
        plt.ylabel("Predictions")

    def PrintAcuracyF1(self):
        print(self.dfF1Acuracy)

    def runModel(self):
        print("model state:"+ str(self.typeCategory))
        if( self.typeCategory == CategoryEnum.OneHot):
            self.runModelOneHot()
        elif (self.typeCategory == CategoryEnum.LabelEcoder):
            self.runModelEncode()
        elif (self.typeCategory == CategoryEnum.LabelEcoderScalar):
            self.runModelEncodeScalar()

    def runModelOneHot(self):

        cols = self.data.columns
        scaler = preprocessing.MinMaxScaler()
        self.data = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(self.data, columns=cols)

        x = self.data[self.colX].to_numpy()
        y = self.data[self.coltarget].to_numpy()

        loso = KFold(n_splits = 10)


        self.mean_fpr = np.linspace(0, 1, 100)

        i=1
        for train_index, test_index in loso.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #Create a Model
            model = SVC(gamma='auto',probability=True)

            #Train the model using the training sets y_pred=clf.predict(X_test)

            probas_ =  model.fit( x_train, y_train ).predict_proba(x_test)
            #Compute ROC curve and area the curve 
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
            #print(tpr)
            self.fpr_tpr_list.append([fpr,tpr])
            self.tprs.append(interp(self.mean_fpr, fpr, tpr))
            self.tprs[-1][0] = 0.0

            roc_auc = auc(fpr, tpr)
            self.aucs.append(roc_auc)

            y_expect = y_test
            y_pred = model.predict( x_test )

            self.ytests += list(y_expect)
            self.ypreds += list(y_pred)

            import pickle

            filerf = 'amazonforest/ic/data/clinvar/02_training/predict_SVC_'+str(i)+'.sav'

            pickle.dump(model, open(filerf, 'wb'))

            i+=1

        dataF1 = {"Acuracy:":[accuracy_score(self.ytests, self.ypreds)],
                  "F1Score:": [f1_score(self.ytests, self.ypreds)],
                  "F1Score (micro):":[f1_score(self.ytests, self.ypreds,average='micro')],
                  "MCC:":[matthews_corrcoef(self.ytests, self.ypreds)]}

        self.dfF1Acuracy = pd.DataFrame(data=dataF1)

        self.dfF1Acuracy.to_csv("amazonforest/ic/data/clinvar/02_training/acuracy_onehot_SVC.csv", index=False)        
        self.plotCM("onehot")
        self.plotRoc ("onehot")

        print("End Model.")


    def runModelEncode(self):


        labelencoder_col = LabelEncoder()
        self.data['FATHMM'] = labelencoder_col.fit_transform(self.data['FATHMM'])
        self.data['LRT_pred'] = labelencoder_col.fit_transform(self.data['LRT_pred'])
        self.data['MutaAss'] = labelencoder_col.fit_transform(self.data['MutaAss'])
        self.data['MutaTaster'] = labelencoder_col.fit_transform(self.data['MutaTaster'])
        self.data['PROVEAN'] = labelencoder_col.fit_transform(self.data['PROVEAN'])
        self.data['Pph2_HDIV'] = labelencoder_col.fit_transform(self.data['Pph2_HDIV'])
        self.data['Pph2_HVAR'] = labelencoder_col.fit_transform(self.data['Pph2_HVAR'])
        self.data['SIFT'] = labelencoder_col.fit_transform(self.data['SIFT'])        

        x = self.data[self.colX].to_numpy()        

        y = 2 == self.data[self.coltarget].to_numpy()

        loso = KFold(n_splits = 10)

        self.mean_fpr = np.linspace(0, 1, 100)

        i=1
        for train_index, test_index in loso.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #Create a Model
            model = SVC(gamma='auto',probability=True)

            probas_ =  model.fit( x_train, y_train ).predict_proba(x_test)            

            #Compute ROC curve and area the curve 
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1]) #,pos_label='1'
            #print(tpr)
            self.fpr_tpr_list.append([fpr,tpr])
            self.tprs.append(interp(self.mean_fpr, fpr, tpr))
            self.tprs[-1][0] = 0.0

            roc_auc = auc(fpr, tpr)
            self.aucs.append(roc_auc)

            y_expect = y_test
            y_pred = model.predict(x_test)

            self.ytests += list(y_expect)
            self.ypreds += list(y_pred)

            i+=1

        dataF1 = {"Acuracy:":[accuracy_score(self.ytests, self.ypreds)],
                  "F1Score:": [f1_score(self.ytests, self.ypreds)],
                  "F1Score (micro):":[f1_score(self.ytests, self.ypreds,average='micro')],
                  "MCC:":[matthews_corrcoef(self.ytests, self.ypreds)]}

        self.dfF1Acuracy = pd.DataFrame(data=dataF1)

        self.dfF1Acuracy.to_csv("amazonforest/ic/data/clinvar/02_training/acuracy_label_SVC.csv",index=False)
        self.plotCM("label")
        self.plotRoc ("label")

        print("End Model.")


    def runModelEncodeScalar(self):

        labelencoder_col = LabelEncoder()
        self.data['FATHMM'] = labelencoder_col.fit_transform(self.data['FATHMM'])
        self.data['LRT_pred'] = labelencoder_col.fit_transform(self.data['LRT_pred'])
        self.data['MutaAss'] = labelencoder_col.fit_transform(self.data['MutaAss'])
        self.data['MutaTaster'] = labelencoder_col.fit_transform(self.data['MutaTaster'])
        self.data['PROVEAN'] = labelencoder_col.fit_transform(self.data['PROVEAN'])
        self.data['Pph2_HDIV'] = labelencoder_col.fit_transform(self.data['Pph2_HDIV'])
        self.data['Pph2_HVAR'] = labelencoder_col.fit_transform(self.data['Pph2_HVAR'])
        self.data['SIFT'] = labelencoder_col.fit_transform(self.data['SIFT'])


        cols = self.data.columns
        scaler = preprocessing.MinMaxScaler()
        self.data = scaler.fit_transform(self.data)
        self.data = pd.DataFrame(self.data, columns=cols)

        x = self.data[self.colX].to_numpy()
        y = self.data[self.coltarget].to_numpy()

        loso = KFold(n_splits = 10)


        self.mean_fpr = np.linspace(0, 1, 100)

        i=1
        for train_index, test_index in loso.split(x):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            #Create a Model
            model = SVC(gamma='auto',probability=True)

            #Train the model using the training sets y_pred=clf.predict(X_test)

            probas_ =  model.fit( x_train, y_train ).predict_proba(x_test)
            #Compute ROC curve and area the curve 
            fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])

            self.fpr_tpr_list.append([fpr,tpr])
            self.tprs.append(interp(self.mean_fpr, fpr, tpr))
            self.tprs[-1][0] = 0.0

            roc_auc = auc(fpr, tpr)
            self.aucs.append(roc_auc)

            y_expect = y_test
            y_pred = model.predict( x_test )

            self.ytests += list(y_expect)
            self.ypreds += list(y_pred)


            i+=1

        dataF1 = {"Acuracy:":[accuracy_score(self.ytests, self.ypreds)],
                  "F1Score:": [f1_score(self.ytests, self.ypreds)],
                  "F1Score (micro):":[f1_score(self.ytests, self.ypreds,average='micro')],
                  "MCC:":[matthews_corrcoef(self.ytests, self.ypreds)]}

        self.dfF1Acuracy = pd.DataFrame(data=dataF1)

        self.dfF1Acuracy.to_csv("amazonforest/ic/data/clinvar/02_training/acuracy_max_SVC.csv",index=False)
        self.plotCM("max")
        self.plotRoc ("max")

        print("End Model.")
