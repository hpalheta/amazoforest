from amazonforest.easy.anotation.EasyService import EasyService
from amazonforest.easy.dal.EasyDal import EasyDal

class EasyFacade:
    def __init__(self):
        self.easyDal = EasyDal()

    def rsToEasySift(self,user_id,easy_id,rsjson,pathsift):
        rslist=[]

        for req in rsjson['easyreq']:
            #print(req['rs'])
            rslist.append(req['rs'])

        if (len(rslist) == 0) :
            return False

        #chamar Easy Service
        easyService = EasyService()

        easyService.run(user_id,easy_id,rslist,None,pathsift)

        return True
    
    def vcfToEasySift(self,user_id,easy_id,vcflistjs,pathsift):

        vcflist=[]

        for req in vcflistjs['easyreq']:
            #print(req['rs'])
            vcflist.append(req['vcf'])

        if (len(vcflist) == 0) :
            return False

        #chamar Easy Service
        easyService = EasyService()

        easyService.run(user_id,easy_id,[],vcflist,pathsift)

        return True

    def get_easy_id(self):
        return self.easyDal.get_easy_id();

    def getpredict_id(self,easyid):
        return self.easyDal.getpredict_id(easyid)
    
    def getrequest_uid(self,userid):
        return self.easyDal.getrequest_uid(userid)

    def getrequest_id(self,easyid):
        return self.easyDal.getrequest_id(easyid)
