#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import collections 
import math
import seaborn as sns
sns.set()
from matplotlib import rcParams
rcParams['figure.figsize'] = 5,4
from matplotlib import pyplot
import statsmodels.api as sm
import time

class ANFIS():
    def __init__(self, function, model,Gamma = 10**6,lam = 0.98, delta=1):
        
        """ 
            A classe ANFIS define um objeto fixo que será treinado para se encaixar a uma função.
        Para inicializalas precisamos definir o número de funções (function) que corresponde ao número de if rules,
        o formato do modelo que desejamos estudar (model), que deve vir como uma lista com [número de inputs, número de outputs]
        o número máximo de gerações permitido (generations),
        e a taxa de aprendizagem (learningRate).
        
        
        Para treinar a classe e assim obter resultados acurados, é preciso realizar o comando train.
        As funções de pertencimento serão gaussianas uniformemente distribuidas, a princípio, sendo modificadas no processo de 
        backpropagation. 
        
        Além do aprendizado do backpropagation, é considerado também o aprendizado por LSE dos parametros de TKS para cada
        set de parâmetros das funções de pertencimento que encontrarmos. 
        
        Apersar de ser adaptativa, já que seus parâmetros são atualizados confome aprendizado de máquina, ele não atualiza os 
        parâmetros após a etapa de treinamento.
        
        
        lam: forgetting factor, usually very close to 1. (online learning on LSE algorithm)
        
        self.num_vars = (self.inputdimension+1)*self.function
        
        delta controls the initial state for the LSE learning.
        
        self.A = delta*np.matrix(np.identity(self.num_vars))
        
        
        """
        self.Gamma = Gamma
        self.function = function
        self.lam = lam
        self.inputdimension = model[0]
        self.outputsdimension = model[1]
        self.Consqparamsdimension = self.inputdimension+1
        self.Wdimension = self.function**self.inputdimension
        self.covarienceMatrixDimension = (self.Consqparamsdimension)*(self.function**self.inputdimension)
        
        self.num_vars = self.covarienceMatrixDimension
        self.delta = delta
        
        self.Deltamu = (np.zeros([self.inputdimension,self.function]))
        self.Deltasig = (np.zeros([self.inputdimension,self.function]))
        self.limit = 0.00001
        self.E = {-1:1}
        
      
    def initialize(self,interval=[-10,10],width = 4):
        
        """ 
            Initialize the arrays for the needed variables.
            
            mu (C) and sig (a) will be defined as an evenly spaced with same lenth argmuents
        """
        
        self.mu = []
        for eachinput in range(self.inputdimension):
            self.mu.append(np.linspace(interval[0],interval[1],num=self.function))
        self.sig = (np.zeros([self.inputdimension,self.function]) + width)
        
            
        self.µ = np.zeros([self.inputdimension,self.function],dtype=np.float64)
        
    
    def gaussian(self,x, cont, cont1):
        
        
        core = (np.power(x - self.mu[cont][cont1], 2) / (np.power(self.sig[cont][cont1], 2)))
        gauss = np.exp(-core)
        if gauss > self.limit:
            self.delmu[self.iteration][cont][cont1] = 2*(x-self.mu[cont][cont1])*np.exp(-core)/np.power(self.sig[cont][cont1], 2)
            self.delsig[self.iteration][cont][cont1] = 2*np.power(x-self.mu[cont][cont1],2)*np.exp(-core)/np.power(self.sig[cont][cont1], 3)

            return gauss
        else:
            self.delmu[self.iteration][cont][cont1] = 0
            self.delsig[self.iteration][cont][cont1] = 0
            return 0
      
    def membershipFunction(self):
        
        """
            Apply the inputs to the membership functions to return an array with dimension (#inputs, #functions).
        
            This way the µ[0] = [X,Y] will be the membership value for the 0th input and 0th rule.
        """
        
        cont=0               
        for eachinput in self.inputs[self.iteration]: # para cada input
            cont1 = 0
            for eachrule in range(self.function): # para cada função no input
                
                self.µ[cont][cont1] = self.gaussian(eachinput,cont,cont1)
                cont1+=1
            cont +=1
        return self.µ
    
    def Tnorm(self,µ,inputdimension,name, W = None,cont=0,cont2=0): #AND operator
        
        """
            Apply the Tnorm function to the µ matrix. Multiplication will be used as "and" operator.
            
            Aqui é feito uma multiplicação de matrizes de n dimensões, sendo n = self.inputdimension (número de inputs).
            
            para isso é necessário que a a dimensão dos vetores µ seja conforme o padrão:
            
            µ[0].shape = [X 1 1 1 ... 1]
            µ[1].shape = [1 X 1 1 ... 1]
            µ[2].shape = [1 1 X 1 ... 1]
            .
            .
            .
            µ[n].shape = [1 1 1 1 ... X]
            
            Sendo X = self.function (número de funções), 
            e que a multiplicação siga a ordem (1. com 2., 1.2. com 3., 1.2.3. com 4. ...). 
            
            isso é feito para realizar a multiplicação matricial das matrizes de multiplas dimensões:
            
            |µ11|                   |µ11µ21  µ11µ22  µ11µ23|
            |µ12| X [µ21 µ22 µ23] = |µ12µ21  µ12µ22  µ12µ23|
            |µ13|                   |µ13µ21  µ13µ22  µ13µ23| 
            
            de n dimensões (neste caso duas), cada uma igual ao número de funções para cada variável. 
        """
        
        
                   
        if type(W)==type(None): #primeira passada
            a,cont2 = self.listador(cont2,inputdimension) 
            W = µ[cont].reshape(a)
            cont+=1
            self.__dict__[name] = W
        if cont<inputdimension:
            a,cont2 = self.listador(cont2,inputdimension)
            
            if cont<inputdimension-1:
                W = np.matmul(µ[cont].reshape(a),W)
            else:
                W = np.matmul(W,µ[cont].reshape(a))
            self.__dict__[name] = W
            cont+=1    
            self.__dict__[name] = self.Tnorm(µ,inputdimension,name,W,cont,cont2)
        return self.__dict__[name]
                
                
    def listador(self,cont2,inputdimension):
        a = []
        for b in range(inputdimension):
            if b == cont2:
                a.append(self.function)
            else:    
                a.append(1)
        cont2+=1
        return a,cont2
        
    def normalization(self):   
        
        """
        Calculo do W barra, bem fácil com numpy
        """
        self.iterWsum[self.iteration] = np.sum(self.W)
        if self.iterWsum[self.iteration] != 0:
            self.Wbarra = self.W/self.iterWsum[self.iteration]
        else:
            self.Wbarra = self.W
        return self.Wbarra
        
    def transform(self,Wbarra):
        
        """ Makes the transformation of Wbarra from a squared [function**(input)] matrix to a complete matrix for all 
        the training data, resulting in a matrix of dimension [N] X [(input+1)*(functions)], in order to find the 
        vector of parameter for the TKS. This is also called as Covarience Matrix.
        
        The final matrix should be: 
        
        transformed[0] = [W00*X0,W11*Y0,...,W00,...,Wn0*X0,Wn0*Y0,...,Wn0]
        transformed[1] = [W01*X1,W01*Y1,...,W01,...,Wn1*X1,Wn1*Y1,...,Wn1]
        .
        .
        .
        transformed[m] = [W0m*Xm,W0m*Ym,...,W0m,...,Wnm*Xm,Wnm*Ym,...,Wnm]
        
        n being the number of inputs and m the nmber of training data pairs.
        Wbarra is an list of arrays.
        THIS FUNCTION SHOULD BE CALLED AFTER THE NORMALIZATION OFF ****ALL**** THE TRAINING DATA PAIRS HAS BEEN PEFORMED.
        
        A submatrix tem um formato semelhante à Tnorm, para poder calcular a matriz de covariância:
        
        [[0.37167694 0.22998752 0.22998752 0.14231245 0.00798868 0.00494326
          0.         0.         0.         0.         0.         0.
          0.         0.         0.         0.         0.         0.        ]
          
         [0.         0.         0.         0.         0.         0.
          0.37167694 0.22998752 0.22998752 0.14231245 0.00798868 0.00494326
          0.         0.         0.         0.         0.         0.        ]
          
         [0.         0.         0.         0.         0.         0.
          0.         0.         0.         0.         0.         0.
          0.37167694 0.22998752 0.22998752 0.14231245 0.00798868 0.00494326]]
          
          for the multiplication:
                   
                    |W1 W2 0  0  0  0  |
          [x,y,1] X |0  0  W1 W2 0  0  | 
                    |0  0  0  0  W1 W2 |
                    
            resultando em uma linha da matriz de covariância para os dados x e y.
            
       """
        matrixfinal = np.zeros([self.maxiter,self.covarienceMatrixDimension])
        datapaircount = 0
        for matrix in Wbarra: # for each data pair
            submatrix = np.zeros([self.Consqparamsdimension,self.covarienceMatrixDimension])
            positioncount = 0
            cont1 = 0
            inputcount = 0
            for value in matrix:
                for eachinput in range(self.Consqparamsdimension):  
                    submatrix[eachinput][positioncount] = value
                    positioncount += 1
            temp = self.inputs[datapaircount].copy()
            temp.append(1)
            matrixfinal[datapaircount] = np.matmul((temp),submatrix) # essa multiplicação é uma forma de alcançar a matrix transformada a partir da matrix de inputs x matriz Wbarra 
            datapaircount += 1
        self.covarienceMatrix = matrixfinal  
        return self.covarienceMatrix

    def initLSETraining(self):
        #num_vars: number of variables including constant
        #lam: forgetting factor, usually very close to 1 not used on current state.
        
       
        self.S = self.Gamma*np.matrix(np.identity(self.num_vars))
        self.A = self.delta*np.matrix(np.identity(self.num_vars))
        self.w = np.matrix(np.zeros(self.num_vars))
        self.w = self.w.reshape(self.w.shape[1],1)
        
        self.lam_inv = self.lam**(-1)
        self.sqrt_lam_inv = math.sqrt(self.lam_inv)
        
        self.a_priori_error = 0
        
        self.num_obs = 0
    
        

    def add_obs(self, x, t):
        '''
        Add the observation x with label t.
        x is a COLUMN vector as a numpy matrix
        t is a real scalar
        '''            

        
        # ((self.S*x)*(x.T*self.S)) me retorna uma matriz cuja diagonal principal é o quadrado dos valores iniciais.
        self.S = self.S - ((self.S*x)*(x.T*self.S))/(1+(x.T*self.S)*x)
        
        # (t - x.T*self.w) is equivalent to the error before the observation
        self.w = self.w + self.S*x*(t - x.T*self.w)
        
    
    
        self.num_obs += 1
        
    def fit(self, X, y):
        '''
        Fit a model to X,y.
        X and y are numpy arrays.
        Individual observations in X should have a prepended 1 for constant coefficient.
        '''
        for i in range(len(X)):
            x = np.transpose(np.matrix(X[i]))
            self.add_obs(x,y[i])


    def get_error(self, generation):
        
        
        return np.mean(abs(self.E[generation]))
    
    def predict(self, x):
        '''
        Predict the value of observation x. x should be a numpy matrix (col vector)
        '''
        return float(np.matmul(self.w.T,x))
    
    
    def showMembershipFunctions(self, initialgen=None, finalgen=None, step=1):
        
        if initialgen == None:
            initialgen = self.generation
        h = lambda x,aa,c: np.exp(-(np.power(x - c, 2) / (np.power(aa, 2))))
        if finalgen == None:
            finalgen = initialgen+1
        for eachinp in range(self.inputdimension):   
            for gen in range(initialgen,finalgen,step):
                try:
                    for aa,c in zip(self.gensig[gen][eachinp],self.genmu[gen][eachinp]):
                        l = []
                        for i in range(self.interval[0]*10,self.interval[1]*10,1):
                            l.append(h(i/10,aa,c))
                        pyplot.plot([i/10 for i in range(self.interval[0]*10,self.interval[1]*10,1)],l)
                except:
                    initialgen -= 1
                    for aa,c in zip(self.gensig[gen][eachinp],self.genmu[gen][eachinp]):
                        l = []
                        for i in range(self.interval[0]*10,self.interval[1]*10,1):
                            l.append(h(i/10,aa,c))
                        pyplot.plot([i/10 for i in range(self.interval[0]*10,self.interval[1]*10,1)],l)
            pyplot.show()
    
    
    def showErrorHistory(self):
        data = {i:np.mean(abs(self.E[i])) for i in range(self.generation+1)}
        pyplot.plot([i for i in range(self.generation+1)],data.values(),"red",label = "RMSE - Dados de treinamento")
        
    
    def LSE(self,w):
        
        """LSE is the method through which the minimum squared errors of the parameters of the TKS method is calculated"""
        self.pred_y = []
        
        if self.training:
            self.initLSETraining()
            for i in range(self.maxiter):
                x = self.covarienceMatrix[i].reshape([self.covarienceMatrixDimension,1]) #transpose
                self.add_obs(np.matrix(x),self.outputs[i])
        else:
            self.w = w
        for i in range(self.maxiter):
            x = self.covarienceMatrix[i].reshape([self.covarienceMatrixDimension,1]) #transpose
            prediction = self.predict(x)
            self.pred_y.append(float(prediction))
        
        if self.training:
            self.errors = np.array(self.outputs) - self.pred_y

            self.E[self.generation] = (np.array(self.outputs) - self.pred_y)
            

        
    def backprop(self):
        """
            Execute the membership function constants tunning via the derivative of the error.
        It's calculated based on the partial derivative of the error for the partial derivative of the constant.
        As we only know the devivative of the constant regarding the function itself, we need to apply the chain rule to find
        the partial derivatives of each layer to the next and the derivative of the error withe the last layer,
        that is easely calculated.
        
        The derivative calculated is:
        for 2 inputs and 2 rules:
        dEp/dparameter = dEp/dFinalLayer*dGauss/dparameter*[ ((pix + qiy +ri)-pred_y)*W1121+((pi'x + qi'y +ri')-pred_y)*W1122]/(sumOfWbarra*µOftheparameter),
        
        It was calculated based on the chain rule on all the layers.
        """
                        
        
        for eachinput in range(self.inputdimension): # para cada input
            for eachrule in range(self.function): # para cada função nesse input
                newMu = 0
                newSig = 0
                for iteration in range(self.maxiter): #considera cada Datapair existente.
                    
                    count = 0
                    newµ = []
                    for i in self.iterµ[iteration]:
                        if count != eachinput:
                            newµ.append(i)
                        else:
                            next
                        count += 1
                        
                    self.newUsefulW = self.Tnorm(newµ,self.inputdimension-1,"newUsefulW") # the -1 is needed to remove the counter of the variable we are diving for
                    
                    divider = (self.iterWsum[iteration]*self.iterµ[iteration][eachinput][eachrule]) #always positive
                    newDivider = (self.iterWsum[iteration]) #always positive
                    if divider > self.limit:
                        
                        temp = self.inputs[iteration].copy()
                        temp.append(1)
                        temp = np.array(temp)#,dtype=np.float64)
                        self.consequentparameters = np.array(np.matmul(self.w.reshape([self.function**self.inputdimension,self.Consqparamsdimension]),temp),dtype=np.float64)#.reshape([self.function for i in range(self.inputdimension)])

                        self.consequentparameters = self.consequentparameters.reshape([self.function for i in range(self.inputdimension)])

                        idle,usefulParameters,idle = lister(self.iterW[iteration], self.consequentparameters-self.pred_y[iteration] 
                                                                    ,eachinput , eachrule,self.inputdimension-1,  
                                                                    np.zeros(self.function**(self.inputdimension-1)),
                                                                    np.zeros(self.function**(self.inputdimension-1)),self.function)
                        newConsequentTerm = np.matmul(self.newUsefulW,usefulParameters) 

                        
                        newMu  +=  -2*(self.errors[iteration])*self.delmu[iteration][eachinput][eachrule]*newConsequentTerm/newDivider

                        newSig += -2*(self.errors[iteration])*self.delsig[iteration][eachinput][eachrule]*newConsequentTerm/newDivider

                    
                self.Deltamu[eachinput][eachrule] = newMu # each variable for each function. should have function * inputs
                self.Deltasig[eachinput][eachrule] = newSig # each variable for each function. should have function * inputs

        n = self.learningRate/abs(np.sum(self.Deltamu)+np.sum(self.Deltasig))

        for eachinput in range(self.inputdimension): # para cada input
            for eachrule in range(self.function): # para cada função nesse input


                self.mu[eachinput][eachrule] += (-1*n)*self.Deltamu[eachinput][eachrule]

                self.sig[eachinput][eachrule] += (-1*n)*self.Deltasig[eachinput][eachrule]

                
        
    
        
        
        

    def train(self, inputs, outputs,interval,width,e=10**-4, retrain=False, generations=500,learningRate=0.98, ResilientProp = True, ResilientPropNred = 2, ResilientPropIncrementValue = 0.01):
        
        """
       
         it's expected that inputs is a list with N dimensions and M repetitions of the input, leaving:
       
         inputs[N,M] as the Nth input on the Mth repetition.
         
         this way, inputs[0] will return all the variables inputs of the first input.
         
         Outputs should correspond one to one to the inputs statments.
        """
        self.generations = generations
        self.learningRate = learningRate
        self.ResilientProp = ResilientProp
        self.training = True
        self.outputs = outputs
        self.interval = interval
        self.width = width
        self.ResilientPropIteractionNeeded = ResilientPropNred
        self.ResilientPropIncrementValue = ResilientPropIncrementValue
        self.execution(inputs,e,self.generations)
        
    def use(self,inputs,generation = None):
        
        """
        
        it's expected that inputs is a list with N dimensions and M repetitions of the input, leaving:
        
        inputs[N,M] as the Nth input on the Mth repetition.
        
        If you want to simulate an specific generation use, specify the generation. If passed as none, the best generation will be used.
        
        """
        
        if generation == None:
            generation = self.Bestgeneration
        
        self.training = False
        
        return self.execution(inputs,10**-28,self.generations+1,generation)        
    
    def test(self, inputs, output):
        
        E = [np.mean(abs(output-np.array(self.use(inputs,i)))) for i in range(self.generation)]
        self.testE = E
        pyplot.plot( [i for i in range(self.generation)],E,"Blue")
        print("Min Error:",np.min(E))
        f = {E[i]:i for i in range(self.generation)}
        self.Besttesterror = f[np.min(E)]
        print("Min Error Generation:",f[np.min(E)])
        
    def execution(self,inputs,e,generations,generation = None):
        
        if not isinstance(inputs,collections.abc.Iterable):
            inputs = [inputs]
        self.inputs = inputs
        self.maxiter = len(self.inputs)
        if self.training:
            reduction = 0
            increase = 0
            self.genw = np.zeros([self.generations,self.num_vars,1])
            dimension = [self.generations,self.maxiter]
            for i in range(self.inputdimension):
                dimension.append(self.function)
            self.genW = np.zeros(dimension)
            self.genmu = np.zeros([self.generations,self.inputdimension,self.function])
            self.gensig = np.zeros([self.generations,self.inputdimension,self.function])
            self.initialize(self.interval,self.width)
            if len(self.outputs)!=self.maxiter:
                raise Exception(f"Output length ( {self.outputs} ) need to be iqual to input length ( {self.maxiter} )")
            self.generation = 0
            self.Bestgeneration = -1
        
            error = 0
        
        end1 = 0
        counter = 3
        while self.generation < generations:
            counter +=1
            ##print("oia")
            start = time.time()
            self.iteration = 0 #controla a iteração do loop de dados de treino de uma geração 
            Wbarra = []

            self.iterµ = np.zeros([self.maxiter,self.inputdimension,self.function],dtype=np.float64)
            dimension = [self.maxiter]
            for i in range(self.inputdimension):
                dimension.append(self.function)
            self.iterW = np.zeros(dimension,dtype=np.float64)
            self.iterWsum = np.zeros([self.maxiter]) # valores da soma de todos os W para cada Datapair
            self.delsig = (np.zeros([self.maxiter,self.inputdimension,self.function]))
            self.delmu = (np.zeros([self.maxiter,self.inputdimension,self.function]))
            if generation != None:
                self.mu = self.genmu[generation]
                self.sig = self.gensig[generation]
            for i in range(self.maxiter):
                µ = (self.membershipFunction())
                self.iterµ[self.iteration] = µ
                self.iterW[self.iteration] = self.Tnorm(self.µ,self.inputdimension,"W")
                Ws = self.normalization().flatten()
                Wbarra.append(Ws)
                self.iteration+=1
            self.transform(Wbarra)
            self.pred_x = [i for i in range(self.maxiter)]
            self.LSE(w = self.genw[generation])
                
            
            
            
            
            if self.training:
                meanerror = self.get_error(self.generation) # np.mean(abs(self.E[self.generation]))
                if meanerror < np.mean(abs(self.E[self.Bestgeneration])) or self.generation == 0:
                    self.Bestgeneration = self.generation
                if meanerror < e:
                    
                    print("Final Error Square:",meanerror  )
                    break
                self.genW[self.generation] = self.iterW
                self.genw[self.generation] = self.w
                self.genmu[self.generation] = self.mu
                self.gensig[self.generation] = self.sig
                end = time.time()
                if counter == 20:
#                     self.showErrorHistory()
                    counter = 0
#                     print(f"Time on Forward pass:{start - end}")
#                     print(f"Time on Backward pass:{end1 - start}")
                # change the learning rate based on the signal of the error. One to one approache
                if self.ResilientProp:
                    if meanerror < self.get_error(self.generation-1):
                        reduction +=1
                        if reduction >= self.ResilientPropIteractionNeeded:
                            self.learningRate += self.learningRate*self.ResilientPropIncrementValue
#                             print("Increasing learning rate to:",self.learningRate,"...")

                        elif increase == 1:
                             self.learningRate -= self.learningRate*0.1
#                              print("Decreasing learning rate to:",self.learningRate,"..." )
                        increase = 0
                    else:
                        reduction = 0
                        increase = 1
                        
                # change the learning rate based on the signal of the error. Jang approach
                        
#                 if self.ResilientProp:
#                     if meanerror < self.get_error(self.generation-1):
#                         reduction +=1
#                         if reduction == 4:
#                             self.learningRate += self.learningRate*0.01
#                             print("Increasing learning rate to:",self.learningRate,"...")
#                             reduction = 0

#                         elif increase == 2:
#                              self.learningRate -= self.learningRate*0.1
#                              print("Decreasing learning rate to:",self.learningRate,"..." )
#                              increase = 0
#                         if reduction != 1:
#                             increase = 0
#                     else:
#                         reduction = 0
#                         increase += 1
#                     print("Generation ",self.generation," mean error:",meanerror)
                
#                 print("Pred_y:", self.pred_y[58])
                if self.generation != self.generations-1:
                    end1 = time.time()
#                     print("==========")
                    self.backprop()
                    
            else:
                generations = 0
                
                    
            self.generation +=1
            
        self.generation -=1
        if self.training:
           
        
            self.showErrorHistory()
            meanerror = self.get_error(self.generation) # np.mean(abs(self.E[self.generation]))
            print("Final generation ",self.generation," mean error:",meanerror)
            
        return self.pred_y
    
            
def lister(function, function1 ,inp , func, count,summ,summ1, num_func, counter1 = 0, Pass = None):
        """
        Get the terms for the derivative of the third layer to the last, given the input and the function that 
        needs to be calculated.
        for the func 1 and inp 1 on a 2 inputs and 2 function basis, we would have: 
        summ = [W1121,W1122], summ1 = [(pix + qiy +ri)-pred_y),(pi'x + qi'y +ri')-pred_y)], or vice-versa
        
        to calculate:
        [ ((pix + qiy +ri)-pred_y)*W1121+((pi'x + qi'y +ri')-pred_y)*W1122], a simple matrix multiplication. 
        
        the parameters sum was already calculated on the consequent parameter variable. 
        =
        The first variable will always be the vertical line projection, while the seccond will always be the honrizontal line.
        Thats why the Pass if exists.
        
        """
        if Pass == None and (inp == 0 or inp == 1):
            #display("inp:",inp)
            #display("func:",func)
            inp = (inp**inp)-inp # 0 will turn to 1 and 1 will turn to 0
            Pass = 0
            #display("---------------------------------------------------------")
            #display("Function1", function1)
            #display("Rule number",func)
            #display("function",function)
            #display("function1",function1)
        if type(function) == np.float64:
            #display("***3***")
            #display("counter1",counter1)
            #display("summ1",summ1)
            #display("function",function)
            #display("function1",function1)
            summ[counter1] = function
            summ1[counter1] = function1
            counter1 +=1
        elif inp == count:
            #display("***1***")
            #display("function",function)
            #display("function1",function1)
            count = -1
            summ,summ1,counter1 = lister(function[func],function1[func],inp,func,count,summ,summ1,num_func,counter1,Pass)
        else:
            
            count-=1
            for eachvariable in range(num_func):
                #display("***2***")
                #display("function",function)
                #display("function1",function1)
                summ,summ1,counter1 = lister(function[eachvariable],function1[eachvariable],inp,func,count,summ,summ1,num_func,counter1,Pass)
#         except IndexError:
#             summ[counter1] = function
#             summ1[counter1] = function1
#             counter1 +=1
        #display("summ1",summ1)
        return summ,summ1,counter1


# In[ ]:




