
import streamlit as st
import pandas as pd
import gpt_2_simple as gpt2
from streamlit_option_menu import option_menu
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np




# PAGE SETUP
st.set_page_config(
    page_title="MyP4K",
    layout="wide",
    page_icon="streamlit_app/assets/p4k_logo.png",

)


# From https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
with open("streamlit_app/style.css") as f:
    st.markdown("""<link href='http://fonts.googleapis.com/css?family=Roboto:400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900italic,900' rel='stylesheet' type='text/css'>""", unsafe_allow_html=True)
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


left_column, center_column,right_column = st.columns([1,3,1])

with left_column:
    st.info("Project using streamlit")
with right_column:
    st.write("##### Authors\nThis tool has been developed by [Emile D. Esmaili](https://github.com/emileDesmaili)")
with center_column:
    st.image("streamlit_app/assets/app_logo.PNG")



st.sidebar.write(f'# Welcome')

page_container = st.sidebar.container()
with page_container:
    page = option_menu("Menu", ["Review Generator", 'Explorer','About'], 
    icons=['reddit','dpad','info'], menu_icon="cast", default_index=0)

# DATA IMPORT
# gpt is cached
sess = gpt2.start_tf_sess()
@st.cache()
def load_gpt():
    gpt2.load_gpt2(sess)
load_gpt()

@st.cache(allow_output_mutation=True)
def load_predictor():
    filename = 'models/score_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model
predictor = load_predictor()

#importing review data with embeddings from BERT
@st.cache()
def load_csv():
    return pd.read_csv('data/raw/pitchfork_data_vec.csv')
df = load_csv()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


if page == 'Review Generator':
#review generator
    st.title('Review Generator')
    with st.form('Review Parameters'):
        prefix = st.text_input('You can type the beginning of the review, or leave it blank')
        length = st.number_input('length of review (number of characters)',1,1500)
        submitted = st.form_submit_button('Generate Review!')
    if submitted:
        with st.spinner('Writing mindblowing, articulate & insightful review...'):
            text = gpt2.generate(sess, prefix=prefix, length=length, return_as_list=True)[0]
        st.success('Well Done, you can almost be a writer for Pitchfork!')
        st.write(text)
        with st.spinner('Consulting all P4K writers to decide on the most accurate score'):
            new_review = model.encode(text)
            score = np.asscalar(predictor.predict(new_review.reshape(1,-1)))
            if score >=8.3:
                st.write('**Best New Music**')
                st.metric('Score:', round(score,1))
            else:
                st.metric('Score:', round(score,1))




if page == 'Explorer':
    st.write(df.head(10))
    
        