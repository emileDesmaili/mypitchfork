
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import plotly.express as px
from transformers import pipeline
import re
from unidecode import unidecode
from streamlit_assets.components import Album 
from jmd_imagescraper.core import * # dont't worry, it's designed to work with import *

# PAGE SETUP
st.set_page_config(
    page_title="MyP4K",
    layout="wide",
    page_icon="streamlit_assets/assets/p4k_logo.png",

)


# From https://discuss.streamlit.io/t/how-to-center-images-latex-header-title-etc/1946/4
with open("streamlit_assets/style.css") as f:
    st.markdown("""<link href='http://fonts.googleapis.com/css?family=Roboto:400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900italic,900' rel='stylesheet' type='text/css'>""", unsafe_allow_html=True)
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


left_column, center_column,right_column = st.columns([1,3,1])

with left_column:
    st.info("Project using streamlit")
with right_column:
    st.write("##### Authors\nThis tool has been developed by [Emile D. Esmaili](https://github.com/emileDesmaili)")
with center_column:
    st.image("streamlit_assets/assets/app_logo.PNG")



st.sidebar.write(f'# Welcome')

page_container = st.sidebar.container()
with page_container:
    page = option_menu("Menu", ["Review Generator", 'Explorer','Review Smart Engine'], 
    icons=['reddit','dpad','info'], menu_icon="cast", default_index=0)

# DATA IMPORT
# gpt is cached

@st.cache(allow_output_mutation=True)
def load_predictor():
    filename = 'models/score_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

#importing review data with embeddings from BERT
@st.cache()
def load_csv():
    return pd.read_csv('data/raw/pitchfork.csv')
@st.cache()
def load_csv_sample():
    return pd.read_csv('data/raw/pitchfork_sample.csv')

@st.cache()
def load_ner_matrix():
    return pd.read_csv('data/raw/ner_matrix.zip')



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    #EmileEsmaili/gpt2-p4k is my model
    model = pipeline('text-generation',"StevenShoemakerNLP/pitchfork")  
 
    return model
model = load_model()

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_w2v():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedder = load_w2v()

### Pages

#review generator
if page == 'Review Generator':

    filename = 'models/score_model.sav'
    predictor = pickle.load(open(filename, 'rb'))

    st.title('Review Generator')

    with st.form('Review Parameters'):
        prefix = st.text_input('You can type the beginning of the review, to specify the artist name for instance, or leave it blank')


        #length = st.number_input('length of review (number of characters)',200,1500)
        submitted = st.form_submit_button('Generate Review!')
    if submitted:
        with st.spinner('Writing mindblowing, articulate & insightful review...'):
            #text = gpt2.generate(sess, prefix=prefix, length=length, return_as_list=True)[0]
            #encoded_input = tokenizer.encode(prefix, return_tensors='pt')
            #output = model.generate(encoded_input, min_length=length)
            #text = tokenizer.decode(output[0], skip_special_tokens=True)
            text = model(prefix,max_length=1023)[0]['generated_text']
            text = text.replace('|EndOfText|','  \n ' )


        st.success('Well Done, you can almost be a writer for Pitchfork!')
        st.markdown(text)
        with st.spinner('Consulting all P4K writers to decide on the most accurate score'):
            new_review = embedder.encode(text)
            score = predictor.predict(new_review.reshape(1,-1)).item()
            if score >=8.2:
                st.write('**Best New Music**')
                st.metric('Score:', round(score,1))
            else:
                st.metric('Score:', round(score,1))


if page == 'Explorer':
    df = load_csv()
    df_sample = load_csv_sample()

    st.title('Pitchfork Review Explorer')
    st.header('Some Facts! (as of early 2019)')
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_score = df['score'].mean()
        st.metric('Average Pitchfork Review Score',round(avg_score,1))
    with col2:
        st.metric('Number of Best New Music awarded',df['bnm'].sum())
    with col3:
        st.metric('Number of perfect 10s',len(df[df['score']==10]))
    col1, col2, col3 = st.columns(3)
    with col1:
        #plotting review histogram
        fig = px.histogram(df['score'], title='Review Score Distribution')
        fig.update_layout(height=400) 
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',}) 
        fig.update_layout(showlegend=False, xaxis_title='Score')
        fig.update_traces(marker_color='firebrick')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        #plotting average score per year
        df_merged = pd.DataFrame(list(zip(df.groupby('release_year').mean()['score'],df.groupby('release_year').sum()['bnm'])),
                     columns=['score','bnm'], index=df.groupby('release_year').mean().index)
        fig = px.scatter(df_merged, y='score',title='Average Score per Year (bubble indicates number of BNMs)',size='bnm')
        fig.update_layout(height=400) 
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',}) 
        fig.update_traces(line=dict(color="firebrick", width=3), marker_color='firebrick')
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title='Average Score')
        st.plotly_chart(fig, use_container_width=True)
    with col3:
        df_merged = df[df['artist']!='Various Artists']
        df_merged = pd.DataFrame(list(zip(df_merged.groupby('artist').count()['score'],df_merged.groupby('artist').sum()['bnm'])),
                     columns=['score','bnm'], index=df_merged.groupby('artist').mean().index)
        df_merged['ratio'] = df_merged['bnm']/df_merged['score']
        fig = px.bar(df_merged['bnm'].nlargest(10), title='Artists with most Best New Music')
        fig.update_layout(height=400) 
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',}) 
        fig.update_layout(showlegend=False, xaxis_title='Score')
        fig.update_traces(marker_color='firebrick')
        st.plotly_chart(fig, use_container_width=True)

    st.header('Sentiment and scores')
    st.markdown('<p>I decided to use pre-trained NLP models to see if sentiment could determine the score of a review. To that effect, I went for two different approaches: </p>'
                '<p>1. A <a href="https://huggingface.co/sshleifer/distilbart-cnn-12-6">Pre-trained BART language model </a> to summarize the review and then score sentiment using a <a href="https://huggingface.co/finiteautomata/bertweet-base-sentiment-analysis">Pre-trained BERT sentiment model </a></p>'
                '<p>2. A sentiment score on the full review, which should be less accurate</p>'
                '<p>The results are clear: <b>regardless of the method, sentiment models are not able to predict the scores</b></p>'
                '<p><b>Bottom Line: It is sometimes hard for us humans to link the tone of the reviewer to the score, it is near impossible for (pretrained) machines 🤖</b></p>'

                , unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(df_sample,x='sentiment_full',y = 'score', trendline='ols',
                        color= df_sample["bnm"].apply(lambda x: "BNM" if x == 1 else "not BNM").astype(str))
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',}) 
        fig.update_layout(showlegend=True, xaxis_title='Sentiment on full review')
        st.plotly_chart(fig)
    with col2:
        fig = px.scatter(df_sample,x='sentiment_sumup',y = 'score', trendline='ols',
                        color= df_sample["bnm"].apply(lambda x: "BNM" if x == 1 else "not BNM").astype(str))
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',}) 
        fig.update_layout(showlegend=True, xaxis_title='Sentiment on summarized review')
        st.plotly_chart(fig)
    
    st.write('**NB**: The sentiment score were calculated from the model output using the standard normalization formula:')
    st.latex(r'''
     score_{composite} = \frac{score_{positive}-score_{negative}}{score_{positive}+score_{negative}+score_{neutral}} 
     ''')


    st.header('Make Your Plots')
    st.write('bnm means Best New Music')
    st.write('Filters')
    filter1, filter2, filter3, filter4, filter5 = st.columns(5)

    with filter1:
        artist = st.multiselect('Artist',(df['artist'].unique()))
    with filter2:
        score = st.slider('Score',0., 10.,(0.,10.),0.5)
    with filter3:
        genre = st.multiselect('Genre',df['genre'].unique())
    with filter4:
        year = st.slider('Release Year',int(df['release_year'].nsmallest(1)),int(df['release_year'].nlargest(1)),
        (int(df['release_year'].nsmallest(1)),int(df['release_year'].nlargest(1))))



    # dataframe filtering
    
    df_filtered = df.loc[(pd.to_numeric(df['release_year']).between(year[0],year[1]))
                            & (pd.to_numeric(df['score']).between(score[0],score[1])) 
                            & df['artist'].isin(artist if artist else df['artist'].unique())
                            & df['genre'].isin(genre if genre else df['genre'].unique())
                        ]

    col1, col2, col3, col4, col5= st.columns(5)
    columns = ['artist','genre','release_year','score','bnm','author']
    if artist:
        columns.remove('artist')
    if genre:
        columns.remove('genre')
    
    with col1:
        y = st.selectbox('Plot',columns)
    with col2:
        grouper = st.selectbox('Data grouping method',('Mean','Sum','Count'))
    with col3:
        x = st.selectbox('Relative to',columns)
    with col4:
        large = st.selectbox('Display top/bottom...',('Top','Bottom','All Values'))
    with col5:
        n = st.number_input('...results',int(5),int(50))
    chart = st.selectbox('Chart Type',('Line','Histogram','Bar', 'Scatter'))
    if x == y:
        st.info(f'You chose {x} twice!')
    else:
        if grouper == 'Mean':
            plot_data = df_filtered.groupby(x).mean()[y]
        if grouper == 'Sum':
            plot_data = df_filtered.groupby(x).sum()[y]
        if grouper =='Count':
            plot_data = df_filtered.groupby(x).count()[y]
        if large =='Top':
            plot_data = plot_data.nlargest(n)
        if large =='Bottom':
            plot_data = plot_data.nsmallest(n)
        if chart =='Line':
            fig = px.line(plot_data)
            fig.update_traces(line=dict(color="firebrick", width=3))
        if chart == 'Bar':
            fig = px.bar(plot_data)
            fig.update_traces(marker_color='firebrick')
        if chart == 'Scatter':
            fig = px.scatter(plot_data)
            fig.update_traces(marker_color='firebrick')
        if chart =='Histogram':
            fig = px.histogram(plot_data)
            fig.update_traces(marker_color='firebrick')
        fig.update_layout(height=600) 
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)','paper_bgcolor': 'rgba(0, 0, 0, 0)',})
        
        st.plotly_chart(fig, use_container_width=True)


if page =='Review Smart Engine':

    df = load_csv()
    ner_matrix = load_ner_matrix()
    if 'engine_df' not in st.session_state:
        st.session_state['engine_df'] = pd.DataFrame([])

    with st.form('Search Reviews'):
        keywords = st.text_input('Keywords in review')
        filter1, filter2, filter3, filter4, filter5 = st.columns(5)
        with filter1:
            artist = st.multiselect('Artist',(df['artist'].unique()))
        with filter2:
            score = st.slider('Score',0., 10.,(0.,10.),0.5)
        with filter3:
            genre = st.multiselect('Genre',df['genre'].unique())
        with filter4:
            year = st.slider('Release Year',int(df['release_year'].nsmallest(1)),int(df['release_year'].nlargest(1)),
            (int(df['release_year'].nsmallest(1)),int(df['release_year'].nlargest(1))))
        
        submitted = st.form_submit_button('Search')
    if submitted:
        # dataframe filtering
        with st.spinner('Searching over 20,000 reviews...'):
            df_filtered = df.loc[(pd.to_numeric(df['release_year']).between(year[0],year[1]))
                            & (pd.to_numeric(df['score']).between(score[0],score[1])) 
                            & df['artist'].isin(artist if artist else df['artist'].unique())
                            & df['genre'].isin(genre if genre else df['genre'].unique())
                            & df['review'].dropna().apply(unidecode).str.contains(keywords,flags=re.IGNORECASE, regex=True)  
                            ]
        st.session_state['engine_df'] = df_filtered.reset_index().head(10)
    

    if st.session_state['engine_df'].empty:
        pass
    else:
        for i in range(len(st.session_state['engine_df'])):
            review = st.session_state['engine_df'].iloc[i]
            
            with st.form(review['album']):
                
                col1, col2 = st.columns([2,3])
                with col1:
                    url = review['link']
                    st.write(f"**{review['artist']}**")
                    link = f"[{review['album']}]" + f"({url})"
                    st.write(link, unsafe_allow_html=True)
                    search_string = str(review['artist']) + ' ' + str(review['album']) + ' cover'
                    search_url = duckduckgo_scrape_urls(search_string, max_results=1)
                    st.image(search_url[0], width=200)

                with col2:
                    st.metric('Score',review['score'])
                    if review['bnm'] ==1:
                        st.write("**Best New Music**")
                    st.write('Release Year: ' + str(int(review['release_year'])))
                    with st.expander('Review'):
                        st.write(review['review'])

                submitted = st.form_submit_button('Get similar reviews')
            if submitted:
                review = Album(review['album'],df=df, matrix=ner_matrix)
                review.display_matches()



                
                



    
    
        