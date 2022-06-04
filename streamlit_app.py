
import streamlit as st
import pandas as pd
import gpt_2_simple as gpt2

sess = gpt2.start_tf_sess()

#import gpt is cached
@st.cache()
def load_gpt():
    gpt2.load_gpt2(sess)
load_gpt()

#importing review data with embeddings from BERT
@st.cache()
def load_csv():
    return pd.read_csv('data/raw/pitchfork_data_vec.csv')
df = load_csv()

#review generator
if st.button('Generate P4K Review'):
    text = gpt2.generate(sess, length=500, return_as_list=True)[0]
    st.success('Well Done, you can almost be a writer for Pitchfork!')
    st.write(text)
    