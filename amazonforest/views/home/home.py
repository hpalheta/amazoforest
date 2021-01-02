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


bp = Blueprint('home',__name__)


@bp.route('/')

def home():
    return redirect('/amazonforest')

    # """Renders the home page."""
    # return render_template(
    #     'home/index.html',
    #     title='',
    #     year=datetime.now().year,
    # )
