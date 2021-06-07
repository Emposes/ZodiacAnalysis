
#Character N-Grams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import sklearn
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#Load dataset
filename = Path(".src/horoscopes.xlsx")

df = pd.read_excel(filename)
df = df4.dropna()

words = ["äôt","don","äì","äôs","Äôre", "äôt", "äôs", "Äì","Äôt", "Äôs" ]

for word in words:
     df['Horoscope'] = df['Horoscope'].str.replace(word, ' ')
     
df4= df4.sample(frac =1, random_state=42)
#Remove stopwords
stop = set(stopwords.words('english'))
df['Horoscope'] = df['Horoscope'].str.lower()
df['text_without_stopwords'] = df['Horoscope'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#Split Dataset
data_train, data_test, y_train, y_true = \
    train_test_split(df['text_without_stopwords'], df['Sign'], test_size=0.2, random_state= 42)

#N-grams parameter
ngrams = 5

#Vectorize words into n-grams
ngram_counter = CountVectorizer(ngram_range=(ngrams, ngrams), analyzer='char')

X_train = ngram_counter.fit_transform(data_train)
X_test  = ngram_counter.transform(data_test)

#LinearSVC
#Classifier
classifier = LinearSVC(max_iter=1000, dual= False)

model = classifier.fit(X_train, y_train)
y_test = model.predict(X_test)

#Confusion Matrix
confmat=confusion_matrix(y_true, y_test)
ticks=np.linspace(1, 12,num=12)
plt.imshow(confmat, interpolation='none')
plt.colorbar()
plt.xticks(ticks,fontsize=6)
plt.yticks(ticks,fontsize=6)
plt.grid(True)
plt.show()

#Accuracy
sklearn.metrics.accuracy_score(y_true, y_test)
