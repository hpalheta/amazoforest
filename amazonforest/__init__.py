"""
The flask application pack
"""
import os

from flask            import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login      import LoginManager
from flask_bcrypt     import Bcrypt
from flask_login import login_user, logout_user, login_required, current_user
from flask_login import LoginManager  #login_manager
from flask_cors import CORS

from werkzeug.utils import secure_filename

basedir = os.path.abspath(os.path.dirname(__file__))
basedir = basedir.replace('/amazonforest','')   

SNPSIFT = basedir + '/biotools/snpEff_4.3'

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


app = Flask(__name__)
app.config['SNPSIFT'] = SNPSIFT
app.config.from_object('amazonforest.configuration.Config')

# enable CORS
CORS(app, resources={r'/*': {'origins': '*'}})

lm = LoginManager() # flask-loginmanager
lm.init_app(app) # init the login manager  #@login_manager.user_loader

db = SQLAlchemy  (app) # flask-sqlalchemy
bc = Bcrypt      (app) # flask-bcrypt

from  amazonforest.views.home import home
from  amazonforest.views.amazonforest import amazonforest
#from  easytosift.views.simple import simple
#from  easytosift.views.api import api

app.register_blueprint(home.bp)
app.register_blueprint(amazonforest.bp)

#app.register_blueprint(easytosift.bpeasy)
#app.register_blueprint(api.bpapi)



