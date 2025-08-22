import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

por_stem=PorterStemmer()

def text_processor(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    p = []
    for i in text:
        if i.isalnum():
            p.append(i)
    text = p[:];
    p.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            p.append(i)

    text = p[:]
    p.clear()
    for i in text:
        p.append(por_stem.stem(i))
    return " ".join(p)


model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

st.title("SMS SAM CLASSIFIER")
ip_sms = st.text_input("Enter the Text")

if st.button("Predict"):
    transformed_msg = text_processor(ip_sms)
    vec_ip=vectorizer.transform([transformed_msg])

    result = model.predict(vec_ip.toarray())[0]

    if result == 1:
        st.header("spam")
    else:
        st.header("not spam")
