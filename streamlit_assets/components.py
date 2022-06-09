import streamlit as st
import pandas as pd 
from sentence_transformers import SentenceTransformer



class Review:
    def __init__(self, text, artist):
        self.text = text
        self.artist = artist


    def word2vec(self):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        text = self.text
        embeddings = model.encode(text)
        self.vec = embeddings



