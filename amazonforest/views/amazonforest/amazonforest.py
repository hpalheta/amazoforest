"""
Routes and views for the flask application.
"""


from flask import Blueprint
from amazonforest.ic.SimpleFacade  import SimpleFacade
from amazonforest.ic.Modeling.SimpleNB import SimpleNB
from amazonforest.ic.Modeling.SimpleRF import SimpleRF
from amazonforest.ic.Modeling.SimpleRFview import SimpleRFview
from amazonforest.ic.Modeling.SimpleRFpredict import SimpleRFpredict
from amazonforest.ic.Modeling.SimpleSVC import SimpleSVC
from amazonforest.ic.Modeling.SimpleMetrics import SimpleMetrics
from amazonforest.ic.predata.SimplePreProcess import SimplePreProcess


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


import matplotlib.pyplot as plt
import io
import base64

from amazonforest  import app
from amazonforest  import  lm, db, bc
from amazonforest.models.amazonforest.models import User
from amazonforest.forms.amazonforest.forms  import LoginForm, RegisterForm

from amazonforest.easy.EasyFacade import EasyFacade
from amazonforest.easy.anotation.EasyService import EasyService

import uuid 


bp = Blueprint('amazonforest',__name__,url_prefix='/amazonforest')

@lm.user_loader
def load_user(user_id):
    #if  current_user.is_authenticated:
    #    return User.query.filter_by(alternative_id=user_id).first()
    #return None
    return User.query.get(int(user_id))


@bp.route('/')
def simple():
    """Renders the home page."""
    if not current_user.is_authenticated:
        username = "hpalheta"
        password = "123321"
        user = User.query.filter_by(user=username).first()
        login_user(user)
                
    
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
            rs = request.form.get('RS', '-1')

            simplefc = SimpleFacade()
            simple = simplefc.getSimpleRFpredict(CategoryEnum.LabelEcoder)

            
            #chamar Easy Service
            easyService = EasyService()
            
            vcflist=[]
            rslist=[]
            pathsift =  app.config['SNPSIFT']
            uid = uuid.uuid1()
            easy_id = str(uid)
            user_id = "hpalheta"



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
                vcf = "chr"+chrom+"\t"+pos+"\t.\t"+ref+"\t"+alt+"\t\t\t"
                vcflist.append(vcf)
                dfpredic = easyService.runamazonforest(user_id,easy_id,[],vcflist,pathsift)
            else :
                rslist.append(rs)
                dfpredic = easyService.runamazonforest(user_id,easy_id,rslist,[],pathsift)

            
            
            if dfpredic.loc[0,"dbNSFP_FATHMM_pred"] != None :
                fathmm = dfpredic.loc[0,"dbNSFP_FATHMM_pred"][0:1]
            else:
                fathmm =""

            if dfpredic.loc[0,"dbNSFP_LRT_pred"] != None :
                lrt = dfpredic.loc[0,"dbNSFP_LRT_pred"][0:1]
                lrt2 = lrt
                if lrt2 == "D":
                    lrt2 = "Z"
            else:
                lrt =""
                lrt2=""

            if dfpredic.loc[0,"dbNSFP_MutationAssessor_pred"] != None :
                mutass = dfpredic.loc[0,"dbNSFP_MutationAssessor_pred"][0:1]
            else:
                mutass =""

            if dfpredic.loc[0,"dbNSFP_MutationTaster_pred"] != None :
                muttaster = dfpredic.loc[0,"dbNSFP_MutationTaster_pred"][0:1]
            else:
                muttaster =""

            if dfpredic.loc[0,"dbNSFP_PROVEAN_pred"] != None :
                provean = dfpredic.loc[0,"dbNSFP_PROVEAN_pred"][0:1]
            else:
                provean =""

            if dfpredic.loc[0,"dbNSFP_Polyphen2_HDIV_pred"] != None :
                pph2_hdiv = dfpredic.loc[0,"dbNSFP_Polyphen2_HDIV_pred"][0:1]
            else:
                pph2_hdiv =""

            if dfpredic.loc[0,"dbNSFP_Polyphen2_HVAR_pred"] != None :
                pph2_hvar = dfpredic.loc[0,"dbNSFP_Polyphen2_HVAR_pred"][0:1]
            else:
                pph2_hvar =""

            if dfpredic.loc[0,"dbNSFP_SIFT_pred"] != None :
                sift = dfpredic.loc[0,"dbNSFP_SIFT_pred"][0:1]
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


            if lrt2 =='Z':
                iCountPre += 1 
                lrt_ds = "Deleterious"
            elif lrt2 =='N':
                iCountPre += 1
                lrt_ds = "Neutral"
            elif lrt2 =='U':
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
                datapre = simple.NewPredict_RF(fathmm,lrt2,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift)
                dspredict = "Benign" #+ str(datapre['PREDICT'][0])
                dsproba = datapre['0'][0]

                if (str(datapre['PREDICT'][0]) == "True"):
                    dspredict = "Pathogenic " # + str(datapre['PREDICT'][0]) 
                    dsproba = datapre['1'][0]

                dsproba = "{:.2f}".format(dsproba)
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
                simple = simplefc.getSimpleRFpredict(CategoryEnum.LabelEcoder)

                datapre = simple.NewPredict_RF(fathmm,lrt,mutass,muttaster,provean,pph2_hdiv,pph2_hvar,sift)
                dspredict = "Benign" #+ str(datapre['PREDICT'][0])
                dsproba = datapre['0'][0]

                if (str(datapre['PREDICT'][0]) == "True"):
                    dspredict = "Pathogenic " # + str(datapre['PREDICT'][0]) 
                    dsproba = datapre['1'][0]

                dsproba = "{:.2f}".format(dsproba)
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

        datasig = getDataClnSig()
        dataMC = getDataClnMC()

        sigalldata = getDataClnSigAjusted()
        climcdata = getCliMC()

        simplefc = SimpleFacade()
        simplepre = simplefc.getPreprocess()
        df = None 

        chrom = "1"
        sigall = "-1"
        mc = "-1"
        limit = "5"
        qtd = "0"



        if request.method == 'GET':
            datap = getCliData("-1","-1","-1","5")
            qtd = getCliDataCount("-1","-1","-1")
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/03Data.html",
                sigalldata = sigalldata,
                climcdata = climcdata,
                data = datap,
                chrom = chrom,
                sigall = sigall,
                cmc = mc,
                limit = limit,
                qtd = qtd,
                datasig = datasig,
                datamc = dataMC
                ))

        if request.method == 'POST':
            chrom = request.form.get('chrom', '1')
            sigall = request.form.get('sigall', '-1')
            mc = request.form.get('mc', '-1')
            limit = request.form.get('limit', '5')

            datap = getCliData(chrom,sigall,mc,limit)
            qtd = getCliDataCount(chrom,sigall,mc)
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/03Data.html",
                sigalldata = sigalldata,
                climcdata = climcdata,
                data = datap,
                chrom = chrom,
                sigall = sigall,
                cmc = mc,
                limit = limit,
                qtd = qtd,
                datasig = datasig,
                datamc = dataMC
                ))

@bp.route('/model')
def model():
    """Renders the nb page."""
    
    simplefc = SimpleFacade()
    simple = simplefc.getSimpleRFview(CategoryEnum.OneHot)

    dfAcu = simple.acuF1toDf()
    imgroc = simple.plotRoc("onehot")
    imgcm = simple.plotCMToImg("onehot")

    return render_template('amazonforest/layouts/auth-default.html',content=render_template(
         "amazonforest/pages/04Model.html"
        ,dataacu = dfAcu.to_html(classes='table align-items-left table-dark table-flush',index=False)
        ,imgroc = imgroc
        ,imgcm = imgcm
        ))

@bp.route('/metrics')
def metrics():
    """Renders the contact page."""
    simplefc = SimpleFacade()    
        
    simple = simplefc.getSimpleMetrics() 

    dfori   = simple.dataSumOri()
    dfpre   = simple.dataPreencode()
    dfmetric   = simple.dataMetric()
    dftrain = simple.dataSumTraining()
    #dfpredictAll = simple.dataSumPred()
    #dfpredictCIPA =simple.dataSumCIPA()
    #dfpredictVUS = simple.dataSumVUS()
    
    return render_template('amazonforest/layouts/auth-default.html',content=render_template(
         "amazonforest/pages/05Metrics.html",
        dataori=dfori.to_html(classes='table align-items-center table-dark table-flush',index=False)
        ,datapre=dfpre.to_html(classes='table align-items-center table-dark table-flush',index=False)
        ,datmetric=dfmetric.to_html(classes='table align-items-center table-dark table-flush',index=False)
        ,datatrain = dftrain.to_html(classes='table align-items-left table-dark table-flush',index=False)
        #,dataall = dfpredictAll.to_html(classes='table align-items-left table-dark table-flush',index=False)
        #,datacipa = dfpredictCIPA.to_html(classes='table align-items-left table-dark table-flush',index=False)
        #,datavus = dfpredictVUS.to_html(classes='table align-items-left table-dark table-flush',index=False)
        ))

@bp.route('/predictvus')
def simplerfpredict2():
    """Renders the contact page."""
    simplefc = SimpleFacade()    
    
    # if (label == '1'):
    #     simple = simplefc.getSimpleSVC(CategoryEnum.LabelEcoder) 
    # elif (label == '2'):
    #     simple = simplefc.getSimpleSVC(CategoryEnum.LabelEcoderScalar) 
    # elif (label == '3'):
    #     simple = simplefc.getSimpleSVCtest(CategoryEnum.OneHot)
        
    simple = simplefc.getSimpleMetrics() 

    #dfori   = simple.dataSumOri()
    #dfpre   = simple.dataPreencode()
    #dfmetric   = simple.dataMetric()
    #dftrain = simple.dataSumTraining()
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
        simple = simplefc.getSimpleRFpredict(CategoryEnum.LabelEcoder)
        
        

        if request.method == 'GET':
            chrom = "1"
            data = simple.dataPredictCutOff(chrom)
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/07RFprob.html",
                data=data.to_html(classes='table align-items-center table-dark table-flush',index=False)
                ,chrom= chrom
                ))

        if request.method == 'POST':
            chrom = request.form.get('chrom', '-1')
            data = simple.dataPredictCutOff(chrom)
            return render_template('amazonforest/layouts/auth-default.html',content=render_template(
                "amazonforest/pages/07RFprob.html",
                data=data.to_html(classes='table align-items-center table-dark table-flush',index=False)
                ,chrom= chrom
                ))



##database

def getCliSigAll():
    try:
        conn = sqlite3.connect("amazonforest/ic/data/simple.db")
        with conn as con:
            cur = con.cursor()
            cur.execute(
                "select idsigall,dssigall from clinvar_clnsigall  order by idsigall")
            rows = cur.fetchall()

            return rows
    except Exception as e:
        print(e)
        con.rollback()

        rows = cur.fetchall()

        return rows

    finally:
        con.close()


def getCliMC():
    try:
        conn = sqlite3.connect("amazonforest/ic/data/simple.db")
        with conn as con:
            cur = con.cursor()
            cur.execute(
                "select idmc,dsmc from clinvar_mc  order by idmc")
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(e)
        con.rollback()
        rows = cur.fetchall()

        return rows

    finally:
        # return render_template("result.html",msg = msg)
        con.close()


def getCliData(chrom,sigall,mc,limit):
    try:
        conn = sqlite3.connect("amazonforest/ic/data/simple.db")
        conn.row_factory = sqlite3.Row

        sql = "select * FROM pdclinvar where 1=1"
        if(chrom != "-1"):
            sql += " and CHROM ='" + chrom+"'"
        
        if(sigall != "-1"):
            sql += " and CLNSIG like'%" + sigall+"%'"

        if(mc != "-1"):
            sql += " and MC like '%" + mc +"%'"

        sql += " limit  " + limit

        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(e)
        con.rollback()
        rows = cur.fetchall()

        return rows

    finally:
        # return render_template("result.html",msg = msg)
        con.close()


def getCliDataInfo(bravaid):
    try:
        conn = sqlite3.connect("amazonforest/ic/data/simple.db")
        conn.row_factory = sqlite3.Row

        sql = "select 	* FROM pdclinvar where BravaID='"+ bravaid+"'"

        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return rows[0]
    except Exception as e:
        print(e)
        con.rollback()
        rows = cur.fetchall()

        return rows

    finally:
        # return render_template("result.html",msg = msg)
        con.close()

def getCliDataCount(chrom,sigall,mc):
    try:
        conn = sqlite3.connect("amazonforest/ic/data/simple.db")
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
        con.rollback()
        return "0"

    finally:
        # return render_template("result.html",msg = msg)
        con.close()


def getDataClnSigAjusted():
    try:
        conn = sqlite3.connect("amazonforest/ic/data/simple.db")
        conn.row_factory = sqlite3.Row

        sql = " SELECT 1 ID,  'BENIG' DSSIGN UNION  SELECT 2 ID,  'PATHO' DSSIGN UNION SELECT 3 ID,  'CIPAT' DSSIGN UNION SELECT 4 ID,  'UNSIG' DSSIGN"
        
        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(e)
        con.rollback()
        rows = cur.fetchall()

        return rows

    finally:
        # return render_template("result.html",msg = msg)
        con.close()


def getDataClnSig():
    try:
        conn = sqlite3.connect("amazonforest/ic/data/simple.db")
        conn.row_factory = sqlite3.Row

        sql = " SELECT  CLNSIg,count(idVCF) Count FROM pdclinvar where CLNSIg <> '' group by CLNSIG"

        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(e)
        con.rollback()
        rows = cur.fetchall()
        return rows

    finally:
        # return render_template("result.html",msg = msg)
        con.close()


def getDataClnMC():
    try:
        conn = sqlite3.connect("amazonforest/ic/data/simple.db")
        conn.row_factory = sqlite3.Row

        sql = " SELECT * FROM clinvar_mc "

        with conn as con:
            cur = con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            return rows
    except Exception as e:
        print(e)
        con.rollback()
        rows = cur.fetchall()

        return rows

    finally:
        # return render_template("result.html",msg = msg)
        con.close()

