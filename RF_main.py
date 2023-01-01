import pandas as pd 
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class node:
    def __init__(self,data):
        self.data = data
        self.father = None
        self.children = []
        self.label = None
    def append(self, obj):
        self.children.append(obj)
    def get_value(self):
        return self.data
    def get_children(self):
        return self.children
    def get_label(self):
        return self.label

def bagging(treeNums):
    seed = 42
    kfold = model_selection.KFold(n_splits=10)
    cart = DecisionTreeClassifier()
    num_trees = treeNums
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, x, y, cv=kfold)
    print('Bagging result with %i trees:\n\t' %treeNums)
    print(results.mean())

def entropyFunc(vec, r):
    countPerclass = dict()
    for i in range(0, r):
        countPerclass[vec[i]]= countPerclass.get(vec[i], 0)+ 1
    numberOfclass = len(countPerclass)
    entPerclass = dict()
    for obj in list(countPerclass.keys()):
        countPerclass[obj] /= r
        # now countPerClass show the percent of each class
        entPerclass[obj] = countPerclass[obj] * (-np.math.log2(countPerclass[obj]))
    entropy = sum(entPerclass.values())
    return entropy , numberOfclass

def check_pure(data):
    len_ = len(data)
    label = y_train[data[0]]
    for obj in data:
        if y_train[obj] != label:
            return 0 #0 means not pure
    return 1  #1 means pure

def computeLabel(data):
    labels = dict()
    for obj in data:
        labels[y_train[obj]] = labels.get(y_train[obj], 0) + 1
    goal , _  = max(labels.items(), key = lambda x:x[1])
    return goal 

def randomFeatures(k,featuresList):
    selected= list()
    len_ = len(featuresList)
    if len_ == 1:
        return featuresList
    else:
        for i in range(0,k):
            random = np.random.randint(0,len_ - 1)
            while featuresList[random] in selected:
                random = np.random.randint(0,len_ - 1)
            selected.append(featuresList[random])
        return selected

def computeEnt(data,Y):
    sampleNumber = len(data)
    ent = dict()
    for i in range(0,sampleNumber):
        temp = data[i]
        ent[Y[temp]] = ent.get(Y[temp],0)+ 1
    classLen = len(ent)
    entPerclass = dict()
    for obj in list(ent.keys()):
        ent[obj] /= sampleNumber
        entPerclass[obj] = ent[obj] * (-np.math.log2(ent[obj]))
    entropy = sum(entPerclass.values())
    return entropy , classLen

def featureSelection(k):
    condidateFeatures = list()
    if k == 1:
        return condidateFeatures.append(k)
    for i in range(0, k):
        random = np.random.randint(1,clm)
        while random in condidateFeatures:
            random = np.random.randint(1,clm)
        condidateFeatures.append(random)
    return condidateFeatures

def computeGain(currentEntropy, feature, n_class, depth, x_):
    data = list()
    if depth == 0:
        data = np.arange(row)
    else:
        data = np.array(x_)
    ent = dict()
    classify = dict()
    vec = np.transpose(x_train)
    vec = vec[feature-1]
    for obj in data:
        temp = classify.get(vec[obj], list())
        temp.append(obj)
        classify[vec[obj]] = temp
    entPervalueOfFeatuers = dict()
    for index in list(classify.keys()):
        temp = classify[index] # now temp is list of index
        n_i = len(temp)
        classLabel = dict()
        for obj in temp:
            lable = y_train[obj]
            classLabel[lable]=classLabel.get(lable,0)+1
        entperClass = dict()
        for obj in list(classLabel.keys()):
            classLabel[obj] /= n_i
            entperClass[obj] = classLabel[obj] *(- np.math.log2(classLabel[obj]))
        entPervalueOfFeatuers[index] = sum(entperClass.values())* n_i/row
    sigmaEnt = sum(entPervalueOfFeatuers.values())
    return currentEntropy - sigmaEnt

def getSplit(features, currentEntropy, n_class, depth, x_):
    GainSplit = dict()
    for feature in features:
        GainSplit[feature] = computeGain(currentEntropy, feature, n_class, depth, x_)
    selected, _ = max(GainSplit.items(), key = lambda x:x[1])
    return selected

def splitData(f, data, depth):
    indexList = list()
    if depth == 0:
        indexList = np.arange(row)
    else:
        indexList = data ##############
    classify = dict()
    for obj in indexList:
        temp = classify.get(x_train[obj][f],list())
        temp.append(obj)
        classify[x_train[obj][f]] = temp
    num = len(classify)
    return classify, num
        
def treeMaker(flag, featureNums, X, Y):
    k = 0
    depth = 0
    if flag == 0 :
        k = np.math.trunc(np.math.log10(featureNums)) # k = log p
    else:
        k = np.math.trunc(np.math.sqrt(featureNums)) # k= sqrt(p)

    condidateFeatures = featureSelection(k)
    currentEntropy , n_class = entropyFunc(Y, row)
    selectedFeature = getSplit(condidateFeatures, currentEntropy, n_class, depth, X)
    condidateFeatures.remove(selectedFeature)
    root = node(selectedFeature)
    root.father = None 
    root.label = None
    newDataset, childNumber = splitData(selectedFeature, X, depth)
    
    for child in list(newDataset.keys()):
        thischild = treeMaker_helper(newDataset[child], condidateFeatures, root, flag, depth, Y)
        if thischild != 0:
            root.append(thischild)
    return root , n_class

def treeMaker_helper(data, features, father, flag, depth, Y): #data and features are list 
    k = len(features)
    checkPure = check_pure(data)
    if k == 0 or checkPure == 1:
        father.label = computeLabel(data)
        return 0
    depth += 1
    if flag == 0 :
        k = np.math.trunc(np.math.log10(k)) # k = log p
    else:
        k = np.math.trunc(np.math.sqrt(k)) # k= sqrt(p)
    condidateFeatures = randomFeatures(k, features)
    currentEntropy, n_class = computeEnt(data,Y)
    selectedFeature = getSplit(condidateFeatures, currentEntropy, n_class, depth, data)
    condidateFeatures.remove(selectedFeature)
    obj = node(selectedFeature)
    obj.father = father
    obj.label = None
    newDataset, childNumber = splitData(selectedFeature, data, depth)
    #child = 0
    #while child < childNumber:
    for child in list(newDataset.keys()):
        thischild = treeMaker_helper(newDataset[child], condidateFeatures, obj, flag, depth, Y)
        if thischild != 0: 
            obj.append(thischild)
    return obj

#____________________________________________________________________________________________
#test phase
#____________________________________________________________________________________________

def predictTest(forest, n_class):
    r, c = x_test.shape
    predict = list()
    for i in range(0, r):
        vec = x_test[i]
        votePerclass = dict()
        for tree in list(forest.values()):
            vote = -100
            vote = getLabels(tree, vec, vote)
            #print(vote)
            votePerclass[vote] = votePerclass.get(vote,0) + 1
        label , _ = max(votePerclass.items(), key = lambda x:x[1])
        predict.append(label)
    #print(predict)
    return predict

def getLabels(thisNode, vec, vote):
    label = thisNode.label
    if thisNode.label is not None:
        vote = thisNode.label
    else:
        f = thisNode.data
        count = vec[f]
        #childerenList = thisNode.children
        #node_ = childerenList[count]
        counter = 0
        node_ = 0
        for i in thisNode.children:
            if count == counter:
                node_ = i
                break
            counter += 1
        vote = getLabels(node_, vec, vote)
    return vote

def randomForest(treeNums, flag, featureNums, X, Y):
    forest = dict()
    for tree in range (1,treeNums+1):
        forest[tree], n_class = treeMaker(flag,featureNums, X, Y)
    '''for tree in list(forest.keys()):
        inja = forest[tree]
        print( inja.data)'''
    #now test the accuracy with this treeNums 
    y_predict = predictTest(forest, n_class)
    print('Random Forest result with %i trees:\n\t' %treeNums)
    print(confusion_matrix(y_test,y_predict))  
    print(classification_report(y_test,y_predict))

#dataset = pd.read_table ('D://machine learning_hw//ML_HW#2//Discrete  Datasets//Glass.txt', encoding='ansi')
dataset = pd.read_csv('D://machine learning_hw//ML_HW#2//Discrete  Datasets//Glass.txt',sep = '\t', encoding='ansi')
#print(dataset)

clm = len(dataset.columns)
df = pd.DataFrame(dataset.values , index = dataset.index , columns = np.arange(1,clm+1,1))
#print(df)
#df = df.drop(clm,axis=1)
_ , clm = df.shape
array = df.values
x = array[:,0:clm-1]
y = array[:,clm-1]

#---------------------------------------------
x_train , x_test , y_train , y_test = train_test_split (x , y ,test_size = .3 , random_state = 42)
print('The shape of dataset is:', df.shape)
row , clm = x_train.shape
flag = 1
L = [11, 21, 31, 41, 51]
for num_trees in L:
    randomForest(num_trees, flag, clm, x_train, y_train)
    bagging(num_trees)
