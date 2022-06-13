import streamlit as st
import pandas as pd 
from sentence_transformers import SentenceTransformer
import numpy as np


class Review:
    def __init__(self, text, artist):
        self.text = text
        self.artist = artist



def n_intersections(a,b):
    return len(list(set(a) & set(b)))

@st.cache()
def ner_matrix():
    #cleaning & formatting NER results
    df = pd.read_csv('data/raw/pitchfork_large.csv').dropna(subset=['review'])
    df['persons'] = df['persons'].str.strip('[]').str.replace("'", '').str.split(',')  
    df['orgs'] = df['orgs'].str.strip('[]').str.replace("'", '').str.split(',')  
    df['entities'] = df['persons'] + df['orgs']
    
    for i in range(len(df)):
        entities = df['entities'].iloc[i]
        clean_entities = []
        for entity in entities:
            clean_entities.append(entity.strip().replace("’s", ""))
        df['entities'].iloc[i] = clean_entities
    #score matrix to measure reviews with similar entities mentioned in them
    score_matrix = np.ones((len(df), len(df)))

    for i in range(len(df)):
        for j in range(i):
            entities_1 = df['entities'].iloc[i]
            entities_2 = df['entities'].iloc[j]
            score = n_intersections(entities_1, entities_2)
            score_matrix[i,j] = score
            score_matrix[j,i] = score
    df = pd.DataFrame(score_matrix, columns=df['album'], index = df['album'])
    df.to_csv('C:/git/mypitchfork/data/raw/ner_matrix.csv')
    return df

@st.cache
def corr_matrix():
    # correlation matrix for embeddings
    df = pd.read_csv('data/raw/pitchfork_large.csv').dropna(subset=['review'])
    embedding_matrix = np.ones((384, len(df)))
    #dumb formatting
    df['vec'] = df['vec'].apply(lambda x: 
                           np.fromstring(
                               x.replace('\n','')
                                .replace('[','')
                                .replace(']','')
                                .replace('  ',' '), sep=' '))
    st.write('done')
  
    embedding_matrix = np.array(df['vec'])
    corr = pd.DataFrame(embedding_matrix,columns=df['album']).corr()
    corr.to_csv('C:/git/mypitchfork/data/raw/corr_matrix.csv')
    return corr


    



        






