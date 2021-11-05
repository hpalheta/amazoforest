import pandas as pd
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, bar, pie, draw, scatter
import os
from amazonforest.ic.Modeling.SimpleEnum import CategoryEnum
from amazonforest.ic.Modeling.SimpleMetrics import SimpleMetrics
from amazonforest.ic.Modeling.SimpleRFpredict import SimpleRFpredict

class SimpleFacade:
    def __init__(self):
        pass

    def getSimpleMetrics(self):
        simple = SimpleMetrics()
        return simple
    
    def getSimpleRFpredict(self):
        simpleRFpredict = SimpleRFpredict()
        return simpleRFpredict
