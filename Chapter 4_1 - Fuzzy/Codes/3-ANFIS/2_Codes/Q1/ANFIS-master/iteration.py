
import numpy as np
import ANFIS
import math
from matplotlib import pyplot
import seaborn as sns
sns.set()
from matplotlib import rcParams
rcParams['figure.figsize'] = 10,7

def Randomize(qnt1,qnt2,int1,int2):
    sample_aa = np.random.uniform(int1,int2,qnt1)

    sample_bb = np.random.uniform(int1,int2,qnt2)
    

    output = []
    inputs = []
    for i,ii in zip(sample_aa,sample_bb):
        inputs.append([i,ii])
        output.append((math.sin(i)/i)*(math.sin(ii)/ii))
        
    return inputs,output

def iteration(i):
    inputs,output = Randomize(121,121,-10,10)
    validationinput, validationoutput = Randomize(150,150,-10,10)
    ANFISDICT = {}
    for index,ii in {"0.001":0.001,"0.005":0.005,"0.01":0.01,"0.05":0.05,"0.1":0.1,"0.15":0.15}.items():
        print("Iteração",i," - ",ii," como incremento")
        anfis = ANFIS.ANFIS(function=7,model=[2,1],Gamma= 10**6,lam = 0.98,delta=1)
        anfis.train(inputs,output,interval = [-10,10], width = 12,e=10**-17,generations=250,learningRate=0.1, ResilientProp = True,ResilientPropNred = 2, ResilientPropIncrementValue = ii)
        anfis.test(validationinput,validationoutput)
        ANFISDICT[index] = [anfis,anfis.Besttesterror]
#         pyplot.show()
    return [i,ANFISDICT] 

def iteration2(i):
    inputs,output = Randomize(121,121,-10,10)
    validationinput, validationoutput = Randomize(150,150,-10,10)
    ANFISDICT = {}
    for ii in range(4,8):
        if ii != 5:
            ANFISDICT2 = {}
            for iii in range(1,5):
                print("Iteração",i," - ",ii," regras - prop ", iii)
                anfis = ANFIS.ANFIS(function=ii,model=[2,1],Gamma= 10**6,lam = 0.98,delta=1)
                anfis.train(inputs,output,interval = [-10,10], width = 12,e=10**-17,generations=250,learningRate=0.1, ResilientProp = True,ResilientPropNred = 2, ResilientPropIncrementValue = 0.01)
                anfis.test(validationinput,validationoutput)
    #             pyplot.show()
                ANFISDICT2[iii] = anfis
            ANFISDICT[ii] = ANFISDICT2
    return [i,ANFISDICT] 


def iteration3(i):
    inputs,output = Randomize(121,121,-10,10)
    validationinput, validationoutput = Randomize(150,150,-10,10)
    ANFISDICT = {}
    for ii in range(i[0],i[1]):
        ANFISDICT2 = {}
        print("Iteração",i," - ",ii," regras - prop ")
        anfis = ANFIS.ANFIS(function=7,model=[2,1],Gamma= 10**6,lam = 0.98,delta=1)
        anfis.train(inputs,output,interval = [-10,10], width = ii,e=10**-17,generations=500,learningRate=0.1, ResilientProp = True,ResilientPropNred = 2, ResilientPropIncrementValue = 0.01)
        anfis.test(validationinput,validationoutput)
#             pyplot.show()
        ANFISDICT[ii] = anfis
    return [i,ANFISDICT] 
