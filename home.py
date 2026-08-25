

import streamlit as st

def home():
    st.Page('home.py', title='Home')
    st.header('Home Page')

pages = {
    "Home": {
        st.Page(home, title='Home')
    },
    "Models":{
        st.Page('app/logistic.py', title='Logistic'),
        st.Page('app/svm.py', title='SVM')
    }
}

pg = st.navigation(pages, position='top')
pg.run()