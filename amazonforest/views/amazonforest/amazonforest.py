"""
Routes and views for the flask application.
"""

import requests
from flask import Blueprint
from amazonforest.ic.SimpleFacade  import SimpleFacade
from amazonforest.ic.Modeling.SimpleMetrics import SimpleMetrics
from amazonforest.ic.Modeling.SimpleEnum import CategoryEnum
from datetime import datetime
from flask import render_template, send_from_directory
from flask import send_file
from flask import request



import math
from PIL import Image, ExifTags

import os
from flask import Blueprint
from dateutil.relativedelta import *
from flask import url_for, redirect
from flask_login  import login_user, logout_user, current_user, login_required
import sqlite3
from werkzeug.utils import secure_filename
import json


import matplotlib.pyplot as plt
import io
import base64

from amazonforest  import app
import uuid 
import pandas as  pd



bp = Blueprint('amazonforest',__name__,url_prefix='/amazonforest')


class APIError(Exception):
    """An API Error Exception"""

    def __init__(self, status):
        self.status = status

    def __str__(self):
        return "APIError: status={}".format(self.status)



FREEAPPS=[{ "name":"Predict by Genomics","id":'bygenomics', "link":'amazonforest/bygenomics', "icon":'fa-flask' , "java":"javascript:ReverseDisplay('bygenomics_more')", "description":"Filter by genomic information or dbSNP rsids." },\
            { "name":"Model", "id":'model',"link":'amazonforest/model' , "icon":'fa-line-chart',"java":"javascript:ReverseDisplay('model_more')", "description":"Model"},\
            { "name":"Predicted VUS Clinvar", "id":'predict',"link":'amazonforest/predict' , "icon":'fa-play-circle',"java":"javascript:ReverseDisplay('predict_more')", "description":"Predict VUS Clinvar."},\
            { "name":"View Data", "id":'data',"link":'amazonforest/data' , "icon":'fa-database',"java":"javascript:ReverseDisplay('data_more')", "description":"Data"},\
            { "name":"Use SnpSift Annotation", "id":'bypredictors',"link":'amazonforest/bypredictors', "icon":'fa-cogs' ,"java":"javascript:ReverseDisplay('iscatterplot_more')", "description":"Fill with results of preditors by snpSift."},\
            { "name":"Data Summary", "id":'metrics',"link":'amazonforest/modeldata' , "icon":'fa-list-alt',"java":"javascript:ReverseDisplay('metrics_more')", "description":"Metrics"},\
            { "name":"Predicted CutOff", "id":'predictvar',"link":'amazonforest/predictvar', "icon":'fa-play-circle-o' ,"java":"javascript:ReverseDisplay('predict_more')", "description":"Predict variants"},\
            { "name":"About", "id":'about',"link":'about', "icon":'fa-info' ,"java":"javascript:ReverseDisplay('about_more')", "description":"About AmazonForest"}]  #,\

@bp.route('/')
def home():
    apps= FREEAPPS
    return render_template('home/index.html',userlogged="yes", apps=apps)


@bp.route('/about')
def about():
    apps= FREEAPPS
    return render_template('home/about.html',userlogged="yes", apps=apps)

@bp.route('/main')
def simple():
    """Renders the home page."""
                
    
    return render_template('amazonforest/layouts/auth-default.html',content=render_template(
            'amazonforest/index.html',
            title='AmazonForest',
            year=datetime.now().year,
        ))
#by genomics
@bp.route('/bygenomics', methods=['GET', 'POST'])
def bygenomics():
        """Renders the contact page."""
        
        if request.method == 'GET':

            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/01Genomics.html",predict=False, annotation = False ,msg="")
            )

        if request.method == 'POST':
            chrom = request.form.get('chrom', '-1')
            pos = request.form.get('POS', '-1')
            ref = request.form.get('ref', '-1')
            alt = request.form.get('alt', '-1')
            rs = request.form.get('RS', '')

            simplefc = SimpleFacade()                        
            amazonpredict =  simplefc.getSimpleRFpredict()
            
            if((rs == "") or (rs == "-1")):
                berro = False
                if(chrom == "-1"):
                    berro = True
                if(pos == ""):
                    berro = True
                if(ref == ""):
                    berro = True
                if(alt == ""):
                    berro = True

                if berro:
                    return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                        "amazonforest/pages/01Genomics.html",
                        msg="* Fields required!",
                        predict = False,
                        annotation = False,
                        chrom = chrom,
                        pos = pos,
                        ref = ref,
                        alt= alt,
                        rs = rs
                        ))
                
            dfpredic = annoteSimple(chrom,pos,ref,alt,rs)    

            if(dfpredic.empty) : 
                dspredict = "AmazonForest does not work!"
                dsproba = ""
                return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                        "amazonforest/pages/01Genomics.html",
                        msg="AmazonForest does not work!",
                        predict = False,
                        annotation = False,
                        chrom = chrom,
                        pos = pos,
                        ref = ref,
                        alt= alt,
                        rs = rs
                        ))
            
            
            
            if dfpredic.loc[0,"FATHMM"] != "" :
                fathmm = dfpredic.loc[0,"FATHMM"][0:1]
            else:
                fathmm =""

            if dfpredic.loc[0,"LRT"] != "" :
                lrt = dfpredic.loc[0,"LRT"][0:1]
            else:
                lrt =""                

            if dfpredic.loc[0,"MutationAssessor"] != "" :
                mutass = dfpredic.loc[0,"MutationAssessor"][0:1]
            else:
                mutass =""

            if dfpredic.loc[0,"MutationTaster"] != "" :
                muttaster = dfpredic.loc[0,"MutationTaster"][0:1]
            else:
                muttaster =""

            if dfpredic.loc[0,"PROVEAN"] != "" :
                provean = dfpredic.loc[0,"PROVEAN"][0:1]
            else:
                provean =""

            if dfpredic.loc[0,"Polyphen2_HDIV"] != "" :
                pph2_hdiv = dfpredic.loc[0,"Polyphen2_HDIV"][0:1]
            else:
                pph2_hdiv =""

            if dfpredic.loc[0,"Polyphen2_HVAR"] != "" :
                pph2_hvar = dfpredic.loc[0,"Polyphen2_HVAR"][0:1]
            else:
                pph2_hvar =""

            if dfpredic.loc[0,"SIFT"] != "" :
                sift = dfpredic.loc[0,"SIFT"][0:1]
            else:
                sift =""

            iCountPre = 0

            if fathmm =='D':
                iCountPre += 1 
                fathmm_ds = "DAMAGING"
            elif fathmm =='T':
                iCountPre += 1
                fathmm_ds = "TOLERATED"
            else:
                fathmm_ds = ""

            if lrt =='D':
                iCountPre += 1 
                lrt_ds = "Deleterious"
            elif lrt =='N':
                iCountPre += 1
                lrt_ds = "Neutral"
            elif lrt =='U':
                iCountPre += 1
                lrt_ds = "Unknown"
            else:
                lrt_ds = ""

            if mutass =='H':
                iCountPre += 1 
                mutass_ds = "high"
            elif mutass =='M':
                iCountPre += 1
                mutass_ds = "medium"
            elif mutass =='L':
                iCountPre += 1
                mutass_ds = "Low"
            elif mutass =='N':
                iCountPre += 1
                mutass_ds = "neutral"
            else:
                mutass_ds = ""

            if muttaster =='A':
                iCountPre += 1 
                muttaster_ds = "disease_causing_automatic"
            elif muttaster =='D':
                iCountPre += 1
                muttaster_ds = "disease_causing"
            elif muttaster =='N':
                iCountPre += 1
                muttaster_ds = "polymorphism"
            elif muttaster =='P':
                iCountPre += 1
                muttaster_ds = "polymorphism_automatic"
            else:
                muttaster_ds = ""

            if provean =='D':
                iCountPre += 1 
                provean_ds = "DAMAGING"
            elif provean =='N':
                iCountPre += 1
                provean_ds = "Neutral"
            else:
                provean_ds = ""

            if pph2_hdiv =='B':
                iCountPre += 1 
                pph2_hdiv_ds = "Benign"
            elif pph2_hdiv =='D':
                iCountPre += 1
                pph2_hdiv_ds = "probably damaging"
            elif pph2_hdiv =='P':
                iCountPre += 1
                pph2_hdiv_ds = "possibly damaging"
            else:
                pph2_hdiv_ds = ""


            if pph2_hvar =='B':
                iCountPre += 1 
                pph2_hvar_ds = "Benign"
            elif pph2_hvar =='D':
                iCountPre += 1
                pph2_hvar_ds = "probably damaging"
            elif pph2_hvar =='P':
                iCountPre += 1
                pph2_hvar_ds = "possibly damaging"
            else:
                pph2_hvar_ds = ""


            if sift =='T':
                iCountPre += 1
                sift_ds = "TOLERATED"
            elif sift =='D':
                iCountPre += 1
                sift_ds = "DAMAGING"
            else:
                sift_ds = ""

            if iCountPre > 7 :
                jsonresult =getRFPredict(fathmm,lrt,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift)

                if((jsonresult != None) and  (len(jsonresult)>0)):
                    if(jsonresult["predict"]=="B"):
                        dspredict = "Benign" 
                        dsproba = jsonresult["benign"]
                    elif(jsonresult["predict"]=="P"):
                        dspredict = "Pathogenic " 
                        dsproba = jsonresult["pathogenic"]
                else:
                    dspredict = "Erro in predict" 
                    dsproba = "0.0"
            else:
                dspredict = "AmazonForest does not work with missing data!"
                dsproba = ""


            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/01Genomics.html",
                predict = True,
                annotation = True,
                chrom = chrom,
                pos = pos,
                ref = ref,
                alt= alt,
                rs = rs,
                fathmm = fathmm_ds,
                lrt = lrt_ds,
                mutass = mutass_ds,
                muttaster = muttaster_ds,
                provean = provean_ds,
                pph2_hdiv = pph2_hdiv_ds,
                pph2_hvar = pph2_hvar_ds,
                sift = sift_ds,
                dspredict = dspredict,
                dsproba = dsproba,
                msg=""
                ))


@bp.route('/bypredictors', methods=['GET', 'POST'])
def bypredictors():
        """Renders the contact page."""
        
        if request.method == 'GET':

            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/02Predictors.html",predict=False)
            )

        if request.method == 'POST':
            fathmm = request.form.get('fathmm', '-1')
            lrt = request.form.get('lrt', '-1')
            mutass = request.form.get('mutass', '-1')
            muttaster = request.form.get('muttaster', '-1')
            provean = request.form.get('provean', '-1')
            pph2_hdiv = request.form.get('pph2_hdiv', '-1')
            pph2_hvar = request.form.get('pph2_hvar', '-1')
            sift = request.form.get('sift', '-1')

            iCountPre = 0

            if fathmm =='D':
                iCountPre += 1
            elif fathmm =='T':
                iCountPre += 1

            if lrt =='Z':
                iCountPre += 1 
            elif lrt =='N':
                iCountPre += 1
            elif lrt =='U':
                iCountPre += 1

            if mutass =='H':
                iCountPre += 1 
            elif mutass =='M':
                iCountPre += 1
            elif mutass =='L':
                iCountPre += 1
            elif mutass =='N':
                iCountPre += 1

            if muttaster =='A':
                iCountPre += 1 
            elif muttaster =='D':
                iCountPre += 1
            elif muttaster =='N':
                iCountPre += 1
            elif muttaster =='P':
                iCountPre += 1
            if provean =='D':
                iCountPre += 1 
            elif provean =='N':
                iCountPre += 1

            if pph2_hdiv =='B':
                iCountPre += 1 
            elif pph2_hdiv =='D':
                iCountPre += 1
            elif pph2_hdiv =='P':
                iCountPre += 1

            if pph2_hvar =='B':
                iCountPre += 1 
            elif pph2_hvar =='D':
                iCountPre += 1
            elif pph2_hvar =='P':
                iCountPre += 1


            if sift =='T':
                iCountPre += 1
            elif sift =='D':
                iCountPre += 1

            if iCountPre > 7 :
                simplefc = SimpleFacade()
                jsonresult =getRFPredict(fathmm,lrt,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift)

                if((jsonresult != None) and  (len(jsonresult)>0)):
                    if(jsonresult["predict"]=="B"):
                        dspredict = "Benign" 
                        dsproba = jsonresult["benign"]
                    elif(jsonresult["predict"]=="P"):
                        dspredict = "Pathogenic " 
                        dsproba = jsonresult["pathogenic"]
                else:
                    dspredict = "Erro in predict" 
                    dsproba = "0.0"
            else:
                dspredict = "AmazonForest does not work with missing data!"
                dsproba = ""


            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/02Predictors.html",
                predict = True,
                fathmm = fathmm,
                lrt = lrt,
                mutass = mutass,
                muttaster = muttaster,
                provean = provean,
                pph2_hdiv = pph2_hdiv,
                pph2_hvar = pph2_hvar,
                sift = sift,
                dspredict = dspredict,
                dsproba = dsproba
                ))


@bp.route('/data' , methods=['GET', 'POST'])
def data():
        """Renders the contact page."""
        #ajustar

        datasig = getDataClnSig()
        dataMC = getDataClnMC()

        sigalldata = getDataClnSigAjusted()
        climcdata = getCliMC()

        simplefc = SimpleFacade()
        df = None 

        chrom = "1"
        sigall = "-1"
        mc = "-1"
        limit = "5"


        if request.method == 'GET':
            datap = getCliData("-1","-1","-1","5")
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/03Data.html",
                sigalldata = sigalldata,
                climcdata = climcdata,
                data = datap,
                chrom = chrom,
                sigall = sigall,
                cmc = mc,
                limit = limit,
                datasig = datasig,
                datamc = dataMC
                ))

        if request.method == 'POST':
            chrom = request.form.get('chrom', '1')
            sigall = request.form.get('sigall', '-1')
            mc = request.form.get('mc', '-1')
            limit = request.form.get('limit', '5')
            datap = getCliData(chrom,sigall,mc,limit)
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/03Data.html",
                sigalldata = sigalldata,
                climcdata = climcdata,
                data = datap,
                chrom = chrom,
                sigall = sigall,
                cmc = mc,
                limit = limit,
                datasig = datasig,
                datamc = dataMC
                ))

@bp.route('/model')
def model():
    """Renders the nb page."""
    #ok
    simplefc = SimpleFacade()    
        
    simple = simplefc.getSimpleMetrics() 

    dfori   = simple.dataSumOri()
    pltori   = simple.dataSumOriPlot()
    dfpre   = simple.dataPreencode()
    pltpre   = simple.dataPreencodePlot()    
    dftrain = simple.dataTrain()
    pltTrain   = simple.dataTrainPlot()
    dfpred = simple.dataPredict()
    pltpred   = simple.dataPredictPlot()

    dfmetric   = None
    return render_template('amazonforest/layouts/auth-default.html',content=render_template(
         "amazonforest/pages/04Model.html",
        dataori=dfori.to_html(classes='stripe',index=False)
        ,pltori = pltori
        ,datapre=dfpre.to_html(classes='stripe',index=False)
        ,pltpre = pltpre
        ))

@bp.route('/modeldata')
def modeldata():
    simplefc = SimpleFacade()    
        
    simple = simplefc.getSimpleMetrics() 

    dfori   = simple.dataSumOri()
    pltori   = simple.dataSumOriPlot()
    dfpre   = simple.dataPreencode()
    pltpre   = simple.dataPreencodePlot()    
    dftrain = simple.dataTrain()
    pltTrain   = simple.dataTrainPlot()
    dfpred = simple.dataPredict()
    pltpred   = simple.dataPredictPlot()

    dfmetric   = None
    return render_template('amazonforest/layouts/auth-default.html',content=render_template(
         "amazonforest/pages/05ModelData.html",
        dataori=dfori.to_html(classes='stripe',index=False)
        ,pltori = pltori
        ,datapre=dfpre.to_html(classes='stripe',index=False)
        ,pltpre = pltpre
        ,datatrain=dftrain.to_html(classes='stripe',index=False)
        ,plttrain = pltTrain
        ,datapred=dfpred.to_html(classes='stripe',index=False)
        ,pltpred = pltpred
        ))


@bp.route('/predictvus', methods=['GET', 'POST'])
def predictvus():
        """Renders the contact page."""

        sigalldata = getDataClnSigAjusted()
        climcdata = getCliMC()

        simplefc = SimpleFacade()
        df = None 

        chrom = "1"
        sigall = "-1"
        mc = "-1"
        limit = "5"


        if request.method == 'GET':
            datap = getCliDataPredictVus("-1","-1","-1","5")
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/06Predictvus.html",
                sigalldata = sigalldata,
                climcdata = climcdata,
                data = datap,
                chrom = chrom,
                sigall = sigall,
                cmc = mc,
                limit = limit,
                ))

        if request.method == 'POST':
            chrom = request.form.get('chrom', '1')
            sigall = request.form.get('sigall', '-1')
            mc = request.form.get('mc', '-1')
            limit = request.form.get('limit', '5')
            datap = getCliDataPredictVus(chrom,sigall,mc,limit)
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/06Predictvus.html",
                sigalldata = sigalldata,
                climcdata = climcdata,
                data = datap,
                chrom = chrom,
                sigall = sigall,
                cmc = mc,
                limit = limit
                ))


@bp.route('/predictvusold')
def simplerfpredict2():
    """Renders the contact page."""
    simplefc = SimpleFacade()
    simple = simplefc.getSimpleMetrics()

    dfpredictAll = simple.dataSumPred()
    dfpredictCIPA =simple.dataSumCIPA()
    dfpredictVUS = simple.dataSumVUS()

    return render_template('amazonforest/layouts/auth-default.html',content=render_template(
         "amazonforest/pages/06Predictvus.html",
         dataall = dfpredictAll.to_html(classes='table align-items-left table-dark table-flush',index=False)
        ,datacipa = dfpredictCIPA.to_html(classes='table align-items-left table-dark table-flush',index=False)
        ,datavus = dfpredictVUS.to_html(classes='table align-items-left table-dark table-flush',index=False)
        ))

@bp.route('/rfprob' , methods=['GET','POST'])
def rfprob():
        """Renders the contact page."""
        simplefc = SimpleFacade()
        df = None 

        chrom = "1"

        if request.method == 'GET':
            datap = getCliDataCutOff("-1")
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/07RFprob.html",
                data = datap,
                chrom = chrom
                ))

        if request.method == 'POST':
            chrom = request.form.get('chrom', '1')
          
            datap = getCliDataCutOff(chrom)
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/07RFprob.html",
                data = datap,
                chrom = chrom
                ))



@bp.route('/team')
def team():
    """Renders the nb page."""
    return render_template('amazonforest/layouts/auth-default.html',content=render_template(
         "amazonforest/pages/08Team.html"
        ))


def getCliSigAll():
    conn = sqlite3.connect("amazonforest/ic/data/simple.db")
    try:        
        with conn as con:
            cur = con.cursor()
            cur.execute(
                "select idsigall,dssigall from clinvar_clnsigall  order by idsigall")
            rows = cur.fetchall()

            return rows
    except Exception as e:
        print(e)
        conn.rollback()
        rows = None
        return rows

    finally:
        conn.close()


def getCliMC():
    conn = sqlite3.connect("amazonforest/ic/data/simple.db")
    try:        
        with conn as con:
            cur = con.cursor()
            cur.execute(
                "select idmc,dsmc from clinvar_mc  order by idmc")
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(e)
        conn.rollback()
        return None

    finally:
        conn.close()


def annoteSimple(chrom,pos,ref,alt,rs):
    srv = "https://wwww2.lghm.ufpa.br"

    basedir = os.path.abspath(os.path.dirname(__file__))
    pathlog = os.path.join(basedir,"logrf.txt")        
    logrf = open(pathlog,"w")
    #logrf.write("anotesimple")

    d = {}
    dfempty = pd.DataFrame(d)
    

    headers = {'content-type': 'application/json'}
    if( len(rs)> 1):        
        payload = {"aforest": [{"rs": "\""+rs+"\""}]}
        resp = requests.post(srv+'/amazonforestapi/annote', data = json.dumps(payload),headers=headers)
    else:
        #"vcf": "chr16\t68867456\t.\tC\tA,G,T\t\t\t"}]
        payload = {"aforest": [{"vcf" : "chr"+str(chrom)+"\t"+str(pos)+"\t.\t"+ref+"\t"+alt+"\t\t\t"}]}
        resp = requests.post(srv+'/amazonforestapi/annote', data = json.dumps(payload),headers=headers)
        
    #logrf.write("akii")
    #print(resp)
    #print(resp.json())
    
    #logrf.write(str(resp.json()))

    #return None

    if(resp==None):
        return dfempty

    logrf.write(resp.json())        

    res_json = json.loads(resp.json().replace("'","\""))
    
    idannote =  res_json["id"]

    if (len(idannote) > 0):
        
        resp = requests.get(srv+'/amazonforestapi/annote/id/'+idannote)
        if resp.status_code != 200:
            # This means something went wrong.
            return dfempty
        
        res_json = json.loads(resp.text.replace("null","\"\""))    

        #jsdict= resp.json()["aforestreq"]
        jsdict= res_json["aforestreq"]
        js = json.dumps(jsdict)
        df = pd.read_json(js)    
        return df

    return dfempty


def getCliData(chrom,sigall,mc,limit):
    if(chrom =="-1"):
        resp = requests.get('http://https://www2.lghm.ufpa.br/amazonforestapi/data/id/1')
        if resp.status_code != 200:
            raise ApiError('GET /data/ {}'.format(resp.status_code))
    else:
        headers = {'content-type': 'application/json'}
        payload = {"aforestreq": [{"chrom" : chrom
                                ,"sigall" : sigall
                                ,"mc" : mc
                                ,"limit" :limit}]}
        resp = requests.post('https://www2.lghm.ufpa.br/amazonforestapi/data', data = json.dumps(payload),headers=headers)


    jsdict= resp.json()["aforestreq"]
    js = json.dumps(jsdict)
    df = pd.read_json(js)    
    return df


    

def getCliDataPredictVus(chrom,sigall,mc,limit):
    if(chrom =="-1"):
        resp = requests.get('https://www2.lghm.ufpa.br/amazonforestapi/datapred/id/1/limit/10')
    else:
        resp = requests.get("https://www2.lghm.ufpa.br/amazonforestapi/datapred/id/"+chrom+"/limit/"+limit)                                
    
    if resp.status_code != 200:
        return None


    jsdict= resp.json()["aforestreq"]
    js = json.dumps(jsdict)
    df = pd.read_json(js)    
    return df

def getCliDataCutOff(chrom):
    
    if(chrom =="-1"):
        resp = requests.get('https://www2.lghm.ufpa.br/amazonforestapi/cutoff/id/1')
        if resp.status_code != 200:
            # This means something went wrong.
            raise ApiError('GET /data/ {}'.format(resp.status_code))
    else:
        resp = requests.get('https://www2.lghm.ufpa.br/amazonforestapi/cutoff/id/'+chrom)
        if resp.status_code != 200:
            # This means something went wrong.
            raise ApiError('GET /data/ {}'.format(resp.status_code))

    jsdict= resp.json()["aforestreq"]
    js = json.dumps(jsdict)
    df = pd.read_json(js)    
    return df    

def getRFPredict(fathmm,lrt,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift):
    conn = sqlite3.connect("amazonforest/ic/data/rfpredict.db")
    try:
        conn = sqlite3.connect("amazonforest/ic/data/rfpredict.db")
        conn.row_factory = sqlite3.Row

        sql = "select * from rfpredict where "
        sql += " fathmm ='" + fathmm+"'"
        sql += " and  lrt ='" + lrt+"'"
        sql += " and  mutass ='" + mutass+"'"
        sql += " and  muttaster ='" + muttaster+"'"
        sql += " and  provean ='" + provean+"'"
        sql += " and  pph2_hdiv ='" + pph2_hdiv+"'"
        sql += " and  pph2_hvar ='" + pph2_hvar+"'"
        sql += " and  sift ='" + sift+"'"

        #print(sql)
        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            #print(rows[0])
            return rows[0]
    except Exception as e:
        print(e)
        conn.rollback()
        return None

    finally:
        # return render_template("result.html",msg = msg)
        con.close()


def getCliDataInfo(bravaid):
    #future developer
    pass
    

def getCliDataCount(chrom,sigall,mc):
    conn = sqlite3.connect("amazonforest/ic/data/simple.db")
    try:
        conn.row_factory = sqlite3.Row

        sql = "select count(BravaID) BravaID FROM pdclinvar where 1=1"
        if(chrom != "-1"):
            sql += " and CHROM ='" + chrom+"'"
        
        if(sigall != "-1"):
            sql += " and CLNSIG like'%" + sigall+"%'"

        if(mc != "-1"):
            sql += " and MC like '%" + mc +"%'"


        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return rows[0][0]
    except Exception as e:
        print(e)
        conn.rollback()
        return "0"

    finally:
        # return render_template("result.html",msg = msg)
        con.close()


def getDataClnSigAjusted():
    conn = sqlite3.connect("amazonforest/ic/data/simple.db")
    try:
        
        conn.row_factory = sqlite3.Row

        sql = " SELECT 1 ID,  'BENIG' DSSIGN UNION  SELECT 2 ID,  'PATHO' DSSIGN UNION SELECT 3 ID,  'CIPAT' DSSIGN UNION SELECT 4 ID,  'UNSIG' DSSIGN"
        
        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        return None

    finally:
        conn.close()


def getDataClnSig():
    conn = sqlite3.connect("amazonforest/ic/data/simple.db")
    try:        
        conn.row_factory = sqlite3.Row
        sql = " SELECT  CLNSIg,count(BravaID) Count FROM pdclinvar where CLNSIg <> '' group by CLNSIG"

        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        return None

    finally:
        conn.close()


def getDataClnMC():
    conn = sqlite3.connect("amazonforest/ic/data/simple.db")
    try:        
        conn.row_factory = sqlite3.Row

        sql = " SELECT * FROM clinvar_mc "

        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        return None

    finally:
        conn.close()





