
from amazonforest.ic.SimpleFacade import SimpleFacade
from amazonforest.ic.Modeling.SimpleEnum import CategoryEnum

def simplenb(opt):
    """Renders the RF page."""
    simplefc = SimpleFacade()
    if (opt == '1'):
        simple = simplefc.getSimpleNB(CategoryEnum.LabelEcoder) 
    elif (opt == '2'):
        simple = simplefc.getSimpleNB(CategoryEnum.LabelEcoderScalar) 
    elif (opt == '3'):
        simple = simplefc.getSimpleNB(CategoryEnum.OneHot) 

    simple.runModel()
    df = simple.data.head()
    dfAcu = simple.AcuF1toDf()

def simplerf(opt):
    """Renders the RF page."""
    simplefc = SimpleFacade()
    if (opt == '1'):
        simple = simplefc.getSimpleRF(CategoryEnum.LabelEcoder) 
    elif (opt == '2'):
        simple = simplefc.getSimpleRF(CategoryEnum.LabelEcoderScalar) 
    elif (opt == '3'):
        simple = simplefc.getSimpleRF(CategoryEnum.OneHot) 

    simple.runModel()
    df = simple.data.head()
    dfAcu = simple.AcuF1toDf()

def simplesvm(opt):
    """Renders the RF page."""
    simplefc = SimpleFacade()
    if (opt == '1'):
        simple = simplefc.getSimpleSVC(CategoryEnum.LabelEcoder) 
    elif (opt == '2'):
        simple = simplefc.getSimpleSVC(CategoryEnum.LabelEcoderScalar) 
    elif (opt == '3'):
        simple = simplefc.getSimpleSVC(CategoryEnum.OneHot) 

    simple.runModel()
    df = simple.data.head()
    dfAcu = simple.AcuF1toDf()



if __name__ == '__main__':
    print("Begin")
    simplenb("1")
    simplenb("2")
    simplenb("3")

    simplerf("1")
    simplerf("2")
    simplerf("3")

    simplesvm("1")
    simplesvm("2")
    simplesvm("3")
    print("End")
