import pandas as pd
import numpy as np

#1
dataset = pd.read_csv('PlayTennis_Train.csv')
print(dataset,"\n")


def entropy(target_col):
    #Entropy Calculation Formula
    elements,counts = np.unique(target_col,return_counts = True)
    print("elements,counts,sum,len(elements)")
    print(elements,",",counts,",",sum(counts),len(elements))
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts))for i in range(len(elements))])
    return entropy

def InfoGain(data,feature,tname):
    print("data,feature,tname")
    print(data,",",feature,",",tname)
    te = entropy(data[tname])
    #print("entropy(data[tname])",te)
    vals,counts= np.unique(data[feature],return_counts=True)
    #InformationGain Calculation Forumla
    we =np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[feature]==vals[i]).dropna()[tname]) for i in range(len(vals))]) #same line
    #print("IGain",we)
    Information_Gain = te - we
    return Information_Gain

f=['Outlook','Temperature','Humidity','Wind']
tname="PlayTennis"

def ID3(data,f,tname):
#data:dataset, f:feature,t-name:target_col    
    if len(np.unique(data[tname])) == 1:
        print("data[tname]\n",data[tname])
        return np.unique(data[tname])[0]
    else:
        item_values = [InfoGain(data,feature,tname) for feature in f]
        print("item_values",item_values)
        print(np.argmax(item_values))
        bfi= np.argmax(item_values)
        #print("f:\n",f)
        #print("f[bfi]:\n",f[bfi])
        bf = f[bfi]
        #print("bfi,bf:\n",bfi,bf)
        tree = {bf:{}}
        f = [i for i in f if i != bf]
        #print("f:",f)
        for value in np.unique(data[bf]):
            #print("data[bf]\n",data[bf])
            #print("value\n",value)
            sub_data = data.where(data[bf] == value).dropna()
            #print("sub_data:\n",sub_data)
            #shrinking of the dataset when a root node is chosen
            subtree = ID3(sub_data,f,tname)
            tree[bf][value] = subtree
            #print("tree[bf][value]",tree[bf][value])
        return(tree)


#pnode=None
tree=ID3(dataset,f,tname)
print("The Decision Tree is:\n",tree)
query=dataset.iloc[:,:].to_dict(orient="records")
#print("UERY",query)
def predict(query,tree):
    for key in list(query.keys()):
        #print("key,qery.keys",key,"\n",query.keys)
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]]
                print("tree[key][query[key]]",tree[key][query[key]])
            except:
                return default
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result


#Testing
#print("\nThe Query to be Checked is:\n",query[12])
result=predict(query[12],tree)
#print("\n\nTesting sample 1:\n",query[12],"PREDICTED =>",result)
result=predict(query[13],tree)
#print("\nTesting sample 2:\n",query[13],"PREDICTED =>",result)
