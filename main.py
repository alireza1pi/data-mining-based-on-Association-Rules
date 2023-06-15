#Import all basic lib


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import time
from mlxtend.frequent_patterns import fpgrowth 



import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
read_file = pd.read_csv (r'tags.txt',delimiter = '\t')
read_file.to_csv (r'tags.csv', index=None)
df=pd.read_csv (r'tags.csv')

c = df.join(df.groupby('Id').nth(0), on='Id', rsuffix='_dup')
cc = c.join(df.groupby('Id').nth(1), on='Id', rsuffix='_dup1').reset_index(drop=True)
ccc = cc.join(df.groupby('Id').nth(2), on='Id', rsuffix='_dup2').reset_index(drop=True)
c1 = ccc.join(df.groupby('Id').nth(3), on='Id', rsuffix='_dup3').reset_index(drop=True)
c2=c1.drop_duplicates()
c4=c2.drop(['Id','Tags'],axis=1)
c4.to_csv (r'final_dataset.csv', index=None)

import csv
dataset = open('final_dataset.csv', 'r')
data1 = csv.reader(dataset, delimiter=",")
data = list(data1)




def strip_list_noempty(mylist):
    newlist = (item.strip() if hasattr(item, 'strip') else item for item in mylist)
    return [item for item in newlist if item != '']
for i in range(0,len(data)):
    data[i]=strip_list_noempty(data[i])

x=data

dataset = x
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
start_time = time.time()
frequent = fpgrowth(df, min_support=0.01, use_colnames=True)
print('Time to find frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))
x1=frequent.nlargest(20,'support')
print(x1)






#2&3
su = frequent.support.unique()#all unique support count#Dictionay storing itemset with same support count key
fredic = {}
for i in range(len(su)):
    inset = list(frequent.loc[frequent.support ==su[i]]['itemsets'])
    fredic[su[i]] = inset #Dictionay storing itemset with  support count <= key
fredic2 = {}
for i in range(len(su)):
    inset2 = list(frequent.loc[frequent.support<=su[i]]['itemsets'])
    fredic2[su[i]] = inset2#Find Closed frequent itemset
start_time = time.time()
cl = []
for index, row in frequent.iterrows():
    isclose = True
    cli = row['itemsets']
    cls = row['support']
    checkset = fredic[cls]
    for i in checkset:
        if (cli!=i):
            if(frozenset.issubset(cli,i)):
                isclose = False
                break
    
    if(isclose):
        cl.append(row['itemsets'])
print('Time to find Close frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))  
    
    
    


start_time = time.time()
ml = []
for index, row in frequent.iterrows():
    isclose = True
    cli = row['itemsets']
    cls = row['support']
    checkset = fredic2[cls]
    for i in checkset:
        if (cli!=i):
            if(frozenset.issubset(cli,i)):
                isclose = False
                break
    
    if(isclose):
        ml.append(row['itemsets'])
print('Time to find Max frequent itemset')
print("--- %s seconds ---" % (time.time() - start_time))

print('Closed itemsets are:',cl)

print('maximal itemsets are:',ml)


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('final_dataset.csv', header = None)

transactions = [] 
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,3)]) #apriori expects products to be in strings(it will be in quotes)
     
from apyori import apriori  
rules = apriori(transactions, min_support = 0.0011, min_confidence = 0.5, min_lift = 3, min_length = 2) #(3*7/7500)

results = list(rules)   
myResults = [list(x) for x in results]  
myRes = []             
for j in range(0, 20):
    myRes.append([list(x) for x in myResults[j][2]])
    
s=pd.DataFrame(myRes)
s




