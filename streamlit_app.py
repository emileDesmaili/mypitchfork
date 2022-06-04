
import streamlit as st
import pandas as pd
import gpt_2_simple as gpt2
from streamlit_option_menu import option_menu
sess = gpt2.start_tf_sess()




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
@st.cache()
def load_gpt():
    gpt2.load_gpt2(sess)
load_gpt()

#importing review data with embeddings from BERT
@st.cache()
def load_csv():
    return pd.read_csv('data/raw/pitchfork_data_vec.csv')
df = load_csv()


if page == 'Review Generator':
#review generator
    st.title('Review Generator')
    prefix = st.text_input('You can type the beginning of the review, or leave it blank')
    if st.button('Generate P4K Review'):
        text = gpt2.generate(sess, prefix=prefix, length=500, return_as_list=True)[0]
        st.success('Well Done, you can almost be a writer for Pitchfork!')
        st.write(text)
        