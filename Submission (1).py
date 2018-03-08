
import pandas as pd
import numpy as np
import scipy.sparse as sp
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer, WordNetLemmatizer
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.metrics import f1_score


# In[3]:


with open('./data/train.dat','r') as f:
    train = pd.DataFrame(line.split('\t') for line in f.readlines())
train.columns = ['label', 'abstract']


with open('./data/test.dat','r') as f:
    test = pd.DataFrame(line for line in f.readlines())
test.columns = ['abstract']


# In[4]:


data = []
for doc in train['abstract']:
     data.append(doc)
for doc in test['abstract']:
     data.append(doc)

data = np.array(data)


labels = []

for lab in train['label']:
     labels.append(lab)

labels = np.array(labels)


# In[5]:


def preprocess(data):
    cleaned = []
    for abstract in data:
        cleaned.append(cleaner(abstract))
    
    stemmed=[]
    for abstract in cleaned:
        stemmed.append(stemmer(abstract))
        
    lemmatized=[]
    for abstract in stemmed:
        lemmatized.append(lemmatizer(abstract))
        
    lower = lower_case(lemmatized)
    split = split_words(lower)
    return split

def cleaner(abstract):
    clean= re.sub(re.compile('<.*?>'), '', abstract)
    clean = re.sub(re.compile(r'\d.*?\d+'),'', clean)
    clean = re.sub(ur"[^\w\d'\s]+",' ',clean)
    clean = re.sub(r'(^[ \t]+|[ \t]+(?=:))', '', clean, flags=re.M)
    
    return clean

def stemmer(abstract):
    port = PorterStemmer()
    return " ".join([port.stem(i) for i in abstract.split()])

def lemmatizer(abstract):
    wnl = WordNetLemmatizer()
    return " ".join([wnl.lemmatize(i, 'v') for i in abstract.split()])

def lower_case(data):
    return [l.lower() for l in data]

def split_words(data):
    return [l.split() for l in data]


# In[6]:


data = preprocess(data)


# In[7]:


data[0]


# In[8]:


def filterLen(docs, minlen):
    r""" filter out terms that are too short. 
    docs is a list of lists, each inner list is a document represented as a list of words
    minlen is the minimum length of the word to keep
    """
    return [ [t.lower().encode('utf-8') for t in d if len(t) >= minlen ] for d in docs ]
   
data = filterLen(data ,5)


# In[9]:


data[0]


# These our the function which we will use to build csr matrix, normalize them , view csr matrix info. We will pass our data to build a normalize csr_matrix.

# In[10]:


def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat

def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

sparse_matrix = build_matrix(data)
csr_info(sparse_matrix)


sparse_matrix = csr_idf(sparse_matrix, copy=True)
sparse_matrix = csr_l2normalize(sparse_matrix, copy=True)


# In[11]:


def splitData(mat, cls, fold=1, d=10):
    r""" Split the matrix and class info into train and test data using d-fold hold-out
    """
    n = mat.shape[0]
    r = int(np.ceil(n*1.0/d))
    mattr = []
    clstr = []
    # split mat and cls into d folds
    for f in range(d):
        if f+1 != fold:
            mattr.append( mat[f*r: min((f+1)*r, n)] )
            clstr.extend( cls[f*r: min((f+1)*r, n)] )
    # join all fold matrices that are not the test matrix
    train = sp.vstack(mattr, format='csr')
    # extract the test matrix and class values associated with the test rows
    test = mat[(fold-1)*r: min(fold*r, n), :]
    clste = cls[(fold-1)*r: min(fold*r, n)]

    return train, clstr, test, clste


# After building csr matrix, we will divide it into train_mat and test_mat, so that our matrix our of same dimensions , which is required for matrix multiplication.

# In[12]:


train_mat = sparse_matrix[0:14438]
test_mat =  sparse_matrix[14438:]


# Now we will calculate cosine similarity between train_mat and test_mat.Due to machine incompetency, we use batch size of 10 to calculate similarity.

# In[13]:


train, clstr, test, clste = splitData(train_mat, labels)


# In[14]:


def cosine_similarity_n_space(m1, m2, batch_size=10):
    assert m1.shape[1] == m2.shape[1]
    ret = np.ndarray((m1.shape[0], m2.shape[0]))
    for row_i in range(0, int(m1.shape[0] / batch_size) + 1):
        start = row_i * batch_size
        end = min([(row_i + 1) * batch_size, m1.shape[0]])
        if end <= start:
            break 
        rows = m1[start: end]
        sim = cosine_similarity(rows, m2) 
        ret[start: end] = sim
    return ret

cosineSimilarityValue = cosine_similarity_n_space(test_mat,train_mat)


# In[15]:


cosineSimilarityValue.shape


# In[32]:


predictions= []
for row in cosineSimilarityValue:

    
    k=200
    partitioned_row_byindex = np.argpartition(-row, k)  
    similar_index = partitioned_row_byindex[:k]

    
    type1 = 0
    type2 = 0
    type3 = 0
    type4 = 0
    type5 = 0

    
    for index in similar_index:
        
        if(labels[index] == '1'):
            type1+=1
        elif(labels[index] == '2'):
            type2+=1
        elif(labels[index] == '3'):
            type3+=1
        elif(labels[index] == '4'):
            type4+=1
        elif(labels[index] == '5'):
            type5+=1 
     
    vote = max(type1,type2,type3,type4,type5)
    
    if vote == type1:
        predictions.append(1)
    elif vote == type2:
        predictions.append(2)
    elif vote == type3:
        predictions.append(3)
    elif vote == type4:
        predictions.append(4)
    elif vote == type5:
        predictions.append(5)
        
    


# In[33]:


f = open('./format.dat', 'w')
for prediction in predictions:
    print >>f, prediction
f.close()


# In[27]:


labels[0]

