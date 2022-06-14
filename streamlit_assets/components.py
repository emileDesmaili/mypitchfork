import streamlit as st
import pandas as pd 
from sentence_transformers import SentenceTransformer
import numpy as np
from jmd_imagescraper.core import * # dont't worry, it's designed to work with import *


class Album:
    
    def __init__(self, name, df):
        self.df = df
        self.name = name
        self.artist = self.df[self.df['album']==self.name]['artist'].iloc[0]
        self.text = self.df[self.df['album']==self.name]['review'].iloc[0]
        self.score = self.df[self.df['album']==self.name]['score'].iloc[0]
        self.bnm = self.df[self.df['album']==self.name]['bnm'].iloc[0]
        self.year = self.df[self.df['album']==self.name]['release_year'].iloc[0]
        self.url_link = self.df[self.df['album']==self.name]['link'].iloc[0]

    
    def load_ner_matrix(self):
        self.matrix = pd.read_csv('data/raw/ner_matrix.zip',usecols=['album',self.name])

    def scrape_cover(self):
        search_string = str(self.artist) + ' ' + str(self.name) + ' cover'
        search_url = duckduckgo_scrape_urls(search_string, max_results=1)
        self.cover = search_url[0]
    
    def get_matches(self, n=5):
        self.load_ner_matrix()
        self.matches = self.matrix.sort_values(by = self.name, ascending=False)['album'].to_list()[:n]
    
    def display_matches(self, n=5):
        self.get_matches(n=n)
        cols = st.columns(n)
        i=0
        for a, x in enumerate(cols):
            with x:
                match = Album(self.matches[i], self.df)
                match.scrape_cover()
                st.write(f'**{match.artist}**')
                st.write(f'[{match.name}]({match.url_link})')
                st.image(match.cover, width=150)
                i+=1














### NOT USED ########

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


    



        






