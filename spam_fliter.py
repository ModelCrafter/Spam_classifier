import email.parser
import numpy as np
import urllib.request
import tarfile
from pathlib import Path
import email
import email.policy
import re
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator , TransformerMixin
from html import unescape
from scipy.sparse import csr_matrix
import nltk
import urlextract
from sklearn.model_selection import cross_val_score , cross_val_predict
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score , recall_score , accuracy_score , confusion_matrix , f1_score , precision_recall_curve
from sklearn.model_selection import GridSearchCV


#
def fetch_spam_data():
    spam_root = "http://spamassassin.apache.org/old/publiccorpus/"
    ham_url = spam_root + "20030228_easy_ham.tar.bz2"
    spam_url = spam_root + "20030228_spam.tar.bz2"

    spam_path = Path() / "datasets" / "spam"
    spam_path.mkdir(parents=True, exist_ok=True)
    for dir_name, tar_name, url in (("easy_ham", "ham", ham_url),
                                    ("spam", "spam", spam_url)):
        if not (spam_path / dir_name).is_dir():
            path = (spam_path / tar_name).with_suffix(".tar.bz2")
            print("Downloading", path)
            urllib.request.urlretrieve(url, path)
            tar_bz2_file = tarfile.open(path)
            tar_bz2_file.extractall(path=spam_path)
            tar_bz2_file.close()
    return [spam_path / dir_name for dir_name in ("easy_ham", "spam")]



ham_dir , spam_dir = fetch_spam_data()
ham_files = [f for f in sorted(ham_dir.iterdir()) if len(f.name)>20]
spam_files = [f for f in sorted(spam_dir.iterdir()) if len(f.name)>20]

def load_emails(filepath):
    with open(filepath ,'rb') as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
    
ham_emails = [load_emails(f) for f in ham_files]
spam_emails =[load_emails(f) for f in spam_files]


def email_to_stract(email):
    if isinstance(email,str):
        return email
    payload = email.get_payload
    if isinstance(payload , list):
        multipart = ','.join([email_to_stract(sub_email) for sub_email in payload])
        return f'multipart{multipart}'
    else:
        return email.get_content_type()
    
def stract_counter(emails:list):
    stracts = Counter()
    for email in emails:
        stract = email_to_stract(email=email)
        stracts[stract] += 1
    return stracts
        


#
x = np.array(ham_emails + spam_emails ,dtype=object)
y = np.array([0]*len(ham_emails) + [1]*len(spam_emails))
x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 42 , test_size = 0.2)


def html_to_plain_text(html):
    html = None
    text = re.sub(r'<head.*?>.*?</head>' , '' , html , flags=re.M | re.S | re.I)
    text = re.sub(r'<a\s.*?>' , 'HYPERLINK' , text , flags=re.M | re.S | re.I)
    text = re.sub(r'<.*?>','',text , flags=re.M | re.S | re.I)
    text = re.sub(r'(\s*\n)+' , '\n' , text , flags=re.M | re.S | re.I)
    return unescape(text)

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ('text/plain','text\html'):
            continue
        try:
            content = part.get_content()
        except:
            content = str(part.get_payload())
        if ctype == 'text/plain':
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)

#
class GetCounterOfWords(BaseEstimator , TransformerMixin):
    def __init__(self , stemming = True , super_lower = True , remove_numbers = True , remove_urls = True , remove_headers = True , remove_symbols = True):
        self.stemming = stemming
        self.super_lower = super_lower
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_headers = remove_headers
        self.remove_symbols = remove_symbols
    def fit(self , X , Y=None):
        return self
    def transform(self , X , Y= None):
        X_transformed = []
        ur_selector = urlextract.URLExtract()
        stemmer = nltk.PorterStemmer()
        for email in X: 
            text = email_to_text(email=email) or " "
            if self.super_lower:
                text = text.lower()
            if self.remove_urls:
                urls = list(set(ur_selector.find_urls(text)))
                urls.sort(key=lambda url: len(url),reverse=True)
                for url in urls:
                    text = text.replace(url,'URL')
            if self.remove_numbers:
                text = re.sub(r'\d+(?:\.\d)?(?:[eE][+-]?\d+)?','NUMBER', text)
            if self.remove_symbols:
                text = re.sub(r'\W+',' ' , text , flags=re.M)
            word_counts = Counter(text.split())
            
            if self.stemming:
                stemd_counter_words = Counter()
                for word , count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemd_counter_words[stemmed_word] += count
                word_counts = stemd_counter_words
            X_transformed.append(word_counts)
        return np.array(X_transformed)

class WordCounterToVectors(BaseEstimator , TransformerMixin):
    def __init__(self , vocabularay = 1000):
        self.vocabularay = vocabularay
        
    def fit(self , X , Y=None):
        total_count = Counter()
        for word_count in X:
            for word , count in word_count.items():
                total_count[word] += min(count,10)
        most_common = total_count.most_common()[:self.vocabularay]
        self.vocabularay_ = {word: index + 1 for index , (word , count) in enumerate(most_common)}
        return self
    def transform(self , X , Y=None):
        rows = []
        columns = []
        data = []
        for row , word_count in enumerate(X):
            for word , count in word_count.items():
                rows.append(row)
                columns.append(self.vocabularay_.get(word , 0))
                data.append(count)
        return csr_matrix((data , (rows , columns)),shape=(len(X) , self.vocabularay + 1))


#
our_pipe = Pipeline([
    ('count_words' , GetCounterOfWords()),
    ('words_to_vectors', WordCounterToVectors())]
)

data_X = our_pipe.fit_transform(x_train)

data_X_test = our_pipe.fit_transform(x_test)

our_classifier = GradientBoostingClassifier(random_state=42 , n_estimators=300 , max_features='sqrt' , learning_rate = 0.2) # best_param: {n_estimators=300 , max_features='sqrt' , learning_rate = 0.2}


param_grid = {
    'n_estimators':[200,300,400],
    'learning_rate':[0.1  , 0.2 , 0.01 , 0.02],
    'max_features':['sqrt' , 'log2'],
}

# 

grid_class = GridSearchCV(our_classifier , cv=3 , param_grid=param_grid)
grid_class.fit(data_X , y_train)
pred_test = grid_class.best_estimator_.predict(data_X_test)


#

print(f'accuracy -> {accuracy_score(y_test , pred_test )}')
print('_'*50)

print(f'precision -> {precision_score(y_test , pred_test )}')
print('_'*50)

print(f'recall -> {recall_score(y_test , pred_test )}')
print('_'*50)
print(f'f1_score -> {f1_score(y_test , pred_test )}')
print('_'*50)

print('|confusion_matrix')
print(confusion_matrix(y_test , pred_test ))






