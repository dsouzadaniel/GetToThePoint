from sklearn.feature_extraction.text import CountVectorizer

def get_n_most_common_words(texts, n=50):
    vec = CountVectorizer(max_features=n)
    vec.fit(texts)
    return vec.get_feature_names()

x = ['Hello this is the best', 'This is also going to be nice','Is this also the best you can do?']

print(get_n_most_common_words(x,n=3))