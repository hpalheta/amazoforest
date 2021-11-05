"""
Routes and views for the flask application.
"""
from flask import Blueprint
from datetime import datetime
from flask import render_template
from flask import request
from flask import jsonify
from flask import redirect
import math
import sqlite3
 
#import requests

bp = Blueprint('home', __name__)
#<section id="top" class="services bg-primary" style="background-color:rgb(71, 71, 73);">

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
