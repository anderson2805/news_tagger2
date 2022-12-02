import streamlit as st
from src.tag_extractor import get_tags
from src.utils import cleanhtml



st.set_page_config(
    layout="wide",
    page_title="News Tags AI",
    page_icon="🧠",
)
st.title('🧠News Tags AI Generator')

title = st.text_input('Title', placeholder= 'Enter Title of News')
description = st.text_area('Description', placeholder= 'Enter Description of News')

if(st.button('Generate Tags')):
    result = (get_tags(cleanhtml(title + ". " + description)))

    topN = st.multiselect(
        'Top 10 Tags',
        result['TopNTags'],
        result['TopNTags'])

    entities = st.multiselect(
        'Entities Tags',
        result['NERTags'],
        result['NERTags'])

    topics = st.multiselect(
        'Topics Tags',
        result['TopicTags'],
        result['TopicTags'])

    theme = st.multiselect(
        'Theme Tags',
        result['ThemeTags'],
        result['ThemeTags'])
