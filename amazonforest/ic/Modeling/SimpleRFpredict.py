import io
import os
import json
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
import subprocess


class SimpleRFpredict:

    def __init__(self):
        self.basedir = os.path.abspath(os.path.dirname(__file__))
        self.basedir = self.basedir.replace('/Modeling','')   
    
    def newPredictAmazonforest(self,fathmm,lrt,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift):
        
        basedir = os.path.abspath(os.path.dirname(__file__))
        
        #d = {'FATHMM': [fathmm] ,'LRT_pred':[lrt],'MutaAss': [mutass],'MutaTaster':[muttaster],'PROVEAN':[provean],'Pph2_HDIV':[pph2_hdiv],
        #    'Pph2_HVAR':[pph2_hvar],'SIFT':[sift]}



        cmd = "/usr/bin/Rscript"
        opt = '--vanilla'
        pathlog = os.path.join(basedir,"rscript","log.txt")        
        path2Rscript = os.path.join(basedir,"rscript","mp_amazonforest.R")        
        pathdata = os.path.join(basedir,"rscript","data","pdclinvar.ori.train.csv")
        pathmodel = os.path.join(basedir,"rscript","data","RF13.Rdata")
        
        logfile = open(pathlog,"a")

        args = [fathmm,lrt,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift,pathdata,pathmodel]
        
        logfile.write("args-"+str(args)+"\n")
        logfile.write("cmd-"+ str([cmd,path2Rscript]) +"\n")
        
        try:
            pyR = subprocess.Popen([cmd,path2Rscript] + args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True)

            stdout, stderr = pyR.communicate()
        
        
            if(len(stderr)>0):

                logfile.write(stderr)
            
                jsonerro = '{"PREDICT": "E", "Benign": "", "Pathogenic": "0.0"}'
                return json.loads(jsonerro)
        except Exception as e:
            logfile.write("erro except")

        jsn = stdout[5:(len(stdout)-2)].replace("\\","")
        jsonresult = json.loads(jsn)
        return jsonresult

    def newPredictAmazonforestCsvFormat(self,fathmm,lrt,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift):
        
        basedir = os.path.abspath(os.path.dirname(__file__))
        
        #d = {'FATHMM': [fathmm] ,'LRT_pred':[lrt],'MutaAss': [mutass],'MutaTaster':[muttaster],'PROVEAN':[provean],'Pph2_HDIV':[pph2_hdiv],
        #    'Pph2_HVAR':[pph2_hvar],'SIFT':[sift]}

        cmd = "/usr/bin/Rscript"
        opt = '--vanilla'
        pathlog = os.path.join(basedir,"rscript","log.txt")        
        path2Rscript = os.path.join(basedir,"rscript","mp_amazonforest.R")        
        pathdata = os.path.join(basedir,"rscript","data","pdclinvar.ori.train.csv")
        pathmodel = os.path.join(basedir,"rscript","data","RF13.Rdata")
        
        logfile = open(pathlog,"a")

        args = [fathmm,lrt,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift,pathdata,pathmodel]
        
        logfile.write("args-"+str(args)+"\n")
        logfile.write("cmd-"+ str([cmd,path2Rscript]) +"\n")
        
        try:
            pyR = subprocess.Popen([cmd,path2Rscript] + args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True)

            stdout, stderr = pyR.communicate()
        
        
            if(len(stderr)>0):
                logfile.write(stderr)
                return "erro"
        except Exception as e:
            logfile.write("erro except")

        jsn = stdout[5:(len(stdout)-2)].replace("\\","")

        jsonresult = json.loads(jsn)
        
        linha = fathmm +"\t"+lrt+"\t"+mutass+"\t"+muttaster+"\t"+provean+"\t"+pph2_hdiv+"\t"+pph2_hvar+"\t"+sift
        linha += "\t"+jsonresult["PREDICT"]
        linha += "\t"+jsonresult["Benign"]
        linha += "\t"+jsonresult["Pathogenic"]

        return linha

    
        
