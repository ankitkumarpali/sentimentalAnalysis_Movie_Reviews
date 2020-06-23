import requests
from flask import Flask,render_template,request
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

app = Flask('__name__')
reviews_train = []
for line in open('C:\\Users\\ankit\\Downloads\\aclImdb\\full_train.txt', encoding = "utf8"):
    reviews_train.append(line.strip())
    
reviews_test = []
for line in open('C:\\Users\\ankit\\Downloads\\aclImdb\\full_test.txt', encoding = "utf8"):
    reviews_test.append(line.strip())

#print(reviews_test[:2])



REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

#lemmatization
def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    
    return reviews

reviews_train_clean = preprocess_reviews(reviews_train)
reviews_test_clean = preprocess_reviews(reviews_test)

#print(reviews_test_clean[:2])


#use stopwords
reviews_train_clean = remove_stop_words(reviews_train_clean)

#use lemmatization
reviews_train_clean = get_lemmatized_text(reviews_train_clean)

target = [1 if i < 2500 else 0 for i in range(5000)]

#for n gram

##cv = CountVectorizer(binary=True, ngram_range=(1, 2))
##cv.fit(reviews_train_clean)
##X = cv.transform(reviews_train_clean)
##X_test = cv.transform(reviews_test_clean)
##



#for word count
cv = CountVectorizer(binary=False)
cv.fit(reviews_train_clean)
X = cv.transform(reviews_train_clean)
X_test = cv.transform(reviews_test_clean)

X_train, X_val, y_train, y_val = train_test_split(
    X, target, train_size = 0.75
)



accuracy=[]
for c in [0.01, 0.05, 0.25, 0.5, 1]:
    
    lr = LogisticRegression(C=c)
    lr.fit(X_train, y_train)
    accuracy.append((c,accuracy_score(y_val, lr.predict(X_val))))
    #print ("Accuracy for C=%s: %s" % (c, accuracy_score(y_val, lr.predict(X_val))))
    
#     Accuracy for C=0.01: 0.87472
#     Accuracy for C=0.05: 0.88368
#     Accuracy for C=0.25: 0.88016
#     Accuracy for C=0.5: 0.87808
#     Accuracy for C=1: 0.87648


final_model = LogisticRegression(C=0.05)
final_model.fit(X, target)
result = accuracy_score(target, final_model.predict(X_test))
#print ("Final Accuracy: %s" % accuracy_score(target, final_model.predict(X_test)))
# Final Accuracy: 0.88128


feature_to_coef = {
    word: coef for word, coef in zip(cv.get_feature_names(), final_model.coef_[0])
}


@app.route('/',methods=["GET","POST"])

def index():
	positive=[]
	negative=[]
	for best_positive in sorted(
		feature_to_coef.items(), 
		key=lambda x: x[1], 
		reverse=True)[:5]:
		positive.append(best_positive)
    
	for best_negative in sorted(
		feature_to_coef.items(), 
		key=lambda x: x[1])[:5]:
		negative.append(best_negative)
		
	return render_template("ankit.html",positive=positive,negative=negative,result=result,accuracy=accuracy)

#     ('excellent', 0.9288812418118644)
#     ('perfect', 0.7934641227980576)
#     ('great', 0.675040909917553)
#     ('amazing', 0.6160398142631545)
#     ('superb', 0.6063967799425831)
    


#     ('worst', -1.367978497228895)
#     ('waste', -1.1684451288279047)
#     ('awful', -1.0277001734353677)
#     ('poorly', -0.8748317895742782)
#     ('boring', -0.8587249740682945)
