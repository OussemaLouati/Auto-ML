from sklearn.cluster import KMeans
import nltk
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from gensim.models import Word2Vec

class Categorical_Handler():
    """
            This class fix categorical value errors using NLP and Clustering techniques.
    """
    def __init__(self):
        pass

    def text_cleaning(self,data,col_name):
        """
            Apply cleaning techniques on text categorical values.
            Parameters:
             --------
                • data (pandas dataframe)
                • col_name (string): - Refers to the column name.
            Returns:
                • list : The new clean column.
        """
        clean_col = []
        for text in data[col_name]:
            # split into words
            tokens = word_tokenize(text)
            # convert to lower case
            tokens = [w.lower() for w in tokens]
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in tokens]
            # remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]
            # filter out stop words
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            text = words[:]
            clean_col.append(' '.join(text))
        return clean_col

    def encode_embedding(self,data,col_name):
        """
            Apply Word2Vec embedding on tokens for each column value and then apply the mean of the vectors.
            Parameters:
             --------
                • data (pandas dataframe)
                • col_name (string): - Refers to the column name.
            Returns:
                • list : For every column value is an a new value derived from the mean of embeddings of that value.
        """
        clean_col = Categorical_Handler.text_cleaning(self,data,col_name)
        tokens_in_sentence = []
        for tokens in clean_col:
            tokens_in_sentence.append(tokens.split(" "))
        # train model
        model = Word2Vec(tokens_in_sentence, min_count=1)
        X = model[model.wv.vocab]
        embedding_sum = []
        l = []
        result = []
        for sentence in tokens_in_sentence:
            embedding_sum = [0] * X.shape[1]
            for word in sentence:
                if word in model.wv.vocab:
                    embedding_sum = np.add(embedding_sum,model.wv[word])
            l.append(embedding_sum)
        for array in l:
            result.append(np.sum(array)/X.shape[1])
        return result

    def stemming(self,data,col_name):
        """
            Apply PorterStemmer on tokens for each column value.
            Parameters:
             --------
                • data (pandas dataframe)
                • col_name (string): - Refers to the column name.
            Returns:
                • list : Each token in the colum value has been stemmed.
        """
        liste = data[col_name]
        stem = []
        ps = PorterStemmer()
        for val in liste:
            S = ""
            for w in val.split():
                rootWord=ps.stem(w)
                S= S+ rootWord
            stem.append(S)
        return stem

    def lemmetazation(self,data,col_name):
        """
            Apply WordNetLemmatizer on tokens for each column value.
            Parameters:
             --------
                • data (pandas dataframe)
                • col_name (string): - Refers to the column name.
            Returns:
                • list : Each token in the colum value has been stemmed.
        """
        liste = data[col_name]
        lem = []
        lemmatizer = WordNetLemmatizer()
        for val in liste:
            S = ""
            for w in val.split():
                rootWord=lemmatizer.lemmatize(w)
                S= S+ rootWord
            lem.append(S)
        return lem

    def clustering(self,data,col_name,nb_of_clusters=3):
        """
            Apply Kmean clustering on the categorical chosen column.
            Parameters:
             --------
                • data (pandas dataframe)
                • col_name (string): - Refers to the column(s) name.
                • nb_of_clusters (int): - Represent the number of clusters.
            Returns:
                • list : - The cluster attribution for each column value.
        """
        kmeans = KMeans(n_clusters =nb_of_clusters)
        X = data.loc[:,[col_name]]
        kmeans.fit(X)
        return kmeans.fit_predict(X)

    def optimal_cluster_nb(self,X,min_range=1,max_range=10):
        """
            Plot a graph for the elbow method to chose the best K value for Kmeans clustering.
            Parameters:
             --------
                • X (list): - Column(s) of the dataframe.
                • min_range (int): - Minimum range value.
                • max_range (int): - Maximum range value.
        """
        cost =[]
        for i in range(min_range, max_range):
            KM = KMeans(n_clusters = i, max_iter = 500)
            KM.fit(X)
            # calculates squared error
            # for the clustered points
            cost.append(KM.inertia_)
        # plot the cost against K values
        plt.plot(range(min_range, max_range), cost, color ='g', linewidth ='3')
        plt.xlabel("Value of K")
        plt.ylabel("Sqaured Error (Cost)")
        plt.show()
