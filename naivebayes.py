import glob
import re
import collections
from collections import Counter
from nltk.stem import WordNetLemmatizer
filename="stoplist.txt"
stplist=[]
stplist1={}
temp1=[]
bayes={}
bayes4={}
bayes1=[]
bayes2=[]
bayes3=[]
bayes5={}
vocab={}
arxiv={}
jdm={}
plos={}
lemmatiser=WordNetLemmatizer()
classlabel1="arxiv"
classlabel2="jdm"
classlabel3="plos"

#create vocabulary by removing stop words
def createvocab(train1):
	for i in range(0,len(train1)):
		tempdict={}
		temp1=re.findall(r'\w+',str(train1[i]))
		for j in range(0,len(temp1)):
			temp1[j]=temp1[j].lower()
			temp1[j]=re.sub(r'[^a-zA-Z0-9]',r'',temp1[j])
			temp1[j]=lemmatiser.lemmatize(temp1[j],pos='v')
			temp1[j]=lemmatiser.lemmatize(temp1[j],pos='a')
			temp1[j]=lemmatiser.lemmatize(temp1[j],pos='n')
			temp1[j]=lemmatiser.lemmatize(temp1[j],pos='r')
			if temp1[j] not in stplist1:
				if temp1[j] not in vocab:
					vocab[temp1[j]]=1
					tempdict[temp1[j]]=1
				elif temp1[j] in vocab and temp1[j] not in tempdict:
					vocab[temp1[j]]+=1
					tempdict[temp1[j]]=1

#create training dataset feature matrix
def createfeatmat(vocab1,classlabel):
	count=0
	for f in glob.glob("./articles/"+classlabel+"/*.txt"):
		tempdict={}
		if count==splithalf:
			break
		with open(f,'r',encoding='cp437') as file:
			temp1=file.read()
			featvocab={}
			tempmat=re.sub(r'[^a-zA-Z0-9 ]',r'',str(temp1)).lower()
			for key in vocab1:
				if key in temp1:
					featvocab[key]=1
				else:
					featvocab[key]=0
			featvocab.update({'classlabel':classlabel})
			bayes[f]=featvocab
			file.close()
		count+=1

#feature matrix for testing dataset
def createtestmat(vocab1,classlabel):
	count=0
	for f in glob.glob("./articles/"+classlabel+"/*.txt"):
		if count>149:
			with open(f,'r',encoding='cp437') as file:
				temp1=file.read()
				temp2=re.findall(r'\w+',str(temp1))
				tempdict={}
				tempdict1={}
				for j in range(0,len(temp2)):
					temp2[j]=re.sub(r'[^a-zA-Z0-9 ]',r'',str(temp2[j])).lower()
					if temp2[j] in vocab1:
						if temp2[j] not in tempdict:
							tempdict[temp2[j]]=1
							tempdict1[temp2[j]]=1
						else:
							tempdict[temp2[j]]+=1
					else:
						if temp2[j] not in tempdict:
							tempdict1[temp2[j]]=0
				bayes4[f]=tempdict
				bayes5[f]=tempdict1
		count+=1

#finding the total of all the three indivisual matrices arxiv,jdm and plos
def addall(x,y,z):
	a=Counter(x)
	b=Counter(y)
	c=Counter(z)
	return a+b+c

#creating the arxiv matrix from the training feature vector
def matarxiv(bay):
	newarxiv={}
	for key in bay:
		if bay[key]['classlabel']=="arxiv":
			for key,val in bay[key].items():
				if val==1:
					if key not in newarxiv:
						newarxiv[key]=1
					else:
						newarxiv[key]+=1
	return newarxiv

#creating the jdm matrix from the training feature vector
def matjdm(bay):
	newjdm={}
	for key in bay:
		if bay[key]['classlabel']=="jdm":
			for key,val in bay[key].items():
				if val==1:
					if key not in newjdm:
						newjdm[key]=1
					else:
						newjdm[key]+=1
	return newjdm

#creating the plos matrix from the training feature vector
def matplos(bay):
	newplos={}
	for key in bay:
		if bay[key]['classlabel']=="plos":
			for key,val in bay[key].items():
				if val==1:
					if key not in newplos:
						newplos[key]=1
					else:
						newplos[key]+=1
	return newplos

#calculate naive bayes probability and calculating accuracy of each class
def bayesprobability(bay1,bay2,a,j,p,t):
	fincount=0
	fincount1=0
	fincount2=0
	fincount3=0
	count1=0
	count2=0
	count3=0
	for key,val in bay1.items():
		prob1=1.0
		prob2=1.0
		prob3=1.0
		for (key1,val1),(key2,val2) in zip(bay1[key].items(),bay2[key].items()):
			if key1 in a:
				if key2==1:
					prob1*=(float)(1-(val1*(a[key1]/t[key1])))
				else:
					prob1*=(float)(val1*(a[key1]/t[key1]))
			if key1 in j:
				if key2==1:
					prob2*=(float)(1-(val1*(j[key1]/t[key1])))
				else:
					prob2*=(float)(val1*(j[key1]/t[key1]))
			if key1 in p:
				if key2==1:
					prob3*=(float)(1-(val1*(p[key1]/t[key1])))
				else:
					prob3*=(float)(val1*(p[key1]/t[key1]))
		final=max(prob1,prob2,prob3)
		key1=key.split("/")
		key2=key1[2].split("\\")
		actual=key2[0]
		if final==prob1:
			print("---------------------------------------------------")
			print("Actual class: "+actual.upper())
			print("Classified class: ARXIV")
			if actual=="arxiv":
				fincount1+=1
				fincount+=1
			count1+=1
		elif final==prob2:
			print("---------------------------------------------------")
			print("Actual class: "+actual.upper())
			print("Classified class: JDM")
			if actual=="jdm":
				fincount2+=1
				fincount+=1
			count2+=1
		else:
			print("---------------------------------------------------")
			print("Actual class: "+actual.upper())
			print("Classified class: PLOS")
			if actual=="plos":
				fincount3+=1
				fincount+=1
			count3+=1
	print("---------------------------------------------------")
	print("Accuracy of Bayes Classifier: "+str(round((float)(fincount/len(bay1))*100,2))+"%")
	print("Accuracy of Bayes Classifier for arxiv is: "+str(round((float)(fincount1/count1)*100,2))+"%")
	print("Accuracy of Bayes Classifier for jdm is: "+str(round((float)(fincount2/count2)*100,2))+"%")
	print("Accuracy of Bayes Classifier for plos is: "+str(round((float)(fincount3/count3)*100,2))+"%")

#reading data from stoplist file
with open(filename) as f:
	for line in f:
		stplist.append(line.split('\''));
for i in range(0,len(stplist)):
	for char in ['\\n',' ',',','\'']:
		stplist[i]=str(stplist[i]).replace(char,'')
		if stplist[i] not in stplist1:
			stplist1[stplist[i]]=1
		else:
			stplist1[stplist[i]]+=1

#reading training half of data 
for f in glob.glob("./articles/arxiv/*.txt"):
	with open(f,'r',encoding='cp437') as file:
		bayes1.append(file.readlines())
		file.close()
splithalf=int(len(bayes1)/2)
train=bayes1[:splithalf]
createvocab(train)

for f in glob.glob("./articles/jdm/*.txt"):
	with open(f,'r',encoding='cp437') as file:
		bayes2.append(file.readlines())
		file.close()
splithalf=int(len(bayes2)/2)
train=bayes2[:splithalf]
createvocab(train)

for f in glob.glob("./articles/plos/*.txt"):
	with open(f,'r',encoding='cp437') as file:
		bayes3.append(file.readlines())
		file.close()
splithalf=int(len(bayes3)/2)
train=bayes3[:splithalf]
createvocab(train)

#removing all non-repeating words from vocabulary
for key,val in vocab.copy().items():
	if val==1:
		del vocab[key]
vocab=collections.OrderedDict(sorted(vocab.items()))

#call to creating feature matrix functions
createfeatmat(vocab,classlabel1)
createfeatmat(vocab,classlabel2)
createfeatmat(vocab,classlabel3)

#calculating indivisual matrix
arxiv=matarxiv(bayes)
jdm=matjdm(bayes)
plos=matplos(bayes)
total=addall(arxiv,jdm,plos)

#creating the test feature vector
createtestmat(vocab,classlabel1)
createtestmat(vocab,classlabel2)
createtestmat(vocab,classlabel3)

#calculating bayes probability using function calls
bayesprobability(bayes4,bayes5,arxiv,jdm,plos,total)