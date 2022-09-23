#!/usr/bin/env Rscript
#' 
#' This is the R Script to perform AmazonForest for functional impact of variant genetics.
#' 
#' The AmazonForest model is a random forest based model trained and tested with eight single 
#' functional predictors
#' 
#' The model is described in https://www.mdpi.com/2079-7737/11/4/538.
#' 
#' Citation: Palheta, H.G.A.; Gonçalves, W.G.; Brito, L.M.; dos Santos, A.R.; dos Reis Matsumoto, M.; 
#'           Ribeiro-dos-Santos, Â.; de Araújo, G.S. AmazonForest: In Silico Metaprediction of Pathogenic 
#'           Variants. Biology 2022, 11, 538. https://doi.org/10.3390/biology11040538
#' 
#' Corresponding authors: Prof. Dr. Gilderlanio Santana de Araujo (gilderlanio [at] gmail.com)
#'                        Ms.C Helber Palheta (hpalheta [at] gmail.com)




#' 0) Required libraries.

suppressPackageStartupMessages(library("ROCR"))
suppressPackageStartupMessages(library("randomForest"))
suppressPackageStartupMessages(library("caret"))


#' 1) The input comprise data for eight functional predictors. We extracted data from SNPSift data for example.
#' Below, a simple way of input to perform a new prediction. 

FATHMM <- 'D'
LRT_pred <- 'N'
MutaAss <- 'L'
MutaTaster <- 'N'
PROVEAN <- 'D'
Pph2_HDIV <- 'D'
Pph2_HVAR <- 'P'
SIFT <- 'D'

predict_values <- c(FATHMM, LRT_pred, MutaAss, MutaTaster, PROVEAN, Pph2_HDIV, Pph2_HVAR, SIFT)
to_predict <- data.frame(FATHMM, LRT_pred, MutaAss, MutaTaster, PROVEAN, Pph2_HDIV, Pph2_HVAR, SIFT)


#'2) Load categorial data for CinVar stored variants. All variantes were annotated for 8 functional predictors.

dataset_original <- read.csv(pathdata)

dataset_original <- transform(
  dataset_original,
  CLNSIG=as.factor(CLNSIG),
  FATHMM=as.factor(FATHMM),
  LRT_pred=as.factor(LRT_pred),
  MutaAss=as.factor(MutaAss),
  MutaTaster=as.factor(MutaTaster),
  PROVEAN=as.factor(PROVEAN),
  Pph2_HDIV=as.factor(Pph2_HDIV),
  Pph2_HVAR=as.factor(Pph2_HVAR),
  SIFT=as.factor(SIFT)
)

dataset_original$MetaSVM <- NULL
dataset_original$CLNSIG <- NULL

# 3) Input data must be encoded for one-hot encoding strategy

to_onehot <- rbind(dataset_original, to_predict)
dummy <- dummyVars(" ~ .", data = to_onehot)
dummy <- data.frame(predict(dummy, newdata = to_onehot)) 

init = dim(dataset_original)[1] + 1 
stop = dim(dummy)[1] + 0
target.to.predict <- dummy[init:stop, ]

#' 4) Load de proposed model

pathmodel <- 'RF13.Rdata'
load(pathmodel)

# 5) Perform prediction with class probabilities
new.predictions <- predict(rf, target.to.predict, type="prob")

# 6) Transform the new predictions in dataframe type.
new.predictions <- as.data.frame(new.predictions)
colnames(new.predictions) <- c("Benign", "Pathogenic")

predict_result = if (new.predictions["Benign"] > new.predictions["Pathogenic"]) "B" else "P"
predict_json  = sprintf('{"PREDICT":"%s", "Benign": "%s", "Pathogenic":"%s"}'
                        , predict_result, new.predictions["Benign"]
                        , new.predictions["Pathogenic"])

# 7) 'A simple way of printing results.
print(predict_json)
