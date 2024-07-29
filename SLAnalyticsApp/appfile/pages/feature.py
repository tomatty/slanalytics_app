import streamlit as st
st.title('Sorry, Under Construction.')

"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
import category_encoders as ce
from datetime import datetime
import time
import os
import io
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'module')))

from visualization import DataVisualization
from preprocessing import Preprocessing

st.sidebar.markdown('---')

# Display a title
st.title('Feature Engineering Before ML')

# file upload

# ベースディレクトリ設定
BASE_UPLOAD_DIR = "appfile/data"

# ベースディレクトリが存在しない場合は作成
if not os.path.exists(BASE_UPLOAD_DIR):
    os.makedirs(BASE_UPLOAD_DIR)

# 今日の日付をサブディレクトリ名として使用
today_dir = os.path.join(BASE_UPLOAD_DIR, datetime.now().strftime("%Y-%m-%d"))

if not os.path.exists(today_dir):
    os.makedirs(today_dir)



# 保存されたファイルのリストを取得
def get_uploaded_files(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename != '.DS_Store':
                files.append(os.path.join(root, filename))
    return files

# 保存されたファイルをリストアップ
uploaded_files = get_uploaded_files(BASE_UPLOAD_DIR)

# 保存されたファイルの選択ボックス
selected_file = st.sidebar.selectbox('Select a file to load or delete', uploaded_files)

if 'df' not in st.session_state:
    st.session_state.df = None

if selected_file:
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        load_button = st.button('Load')
    with col2:
        refresh_button = st.button('Refresh')
    with col3:
        delete_button = st.button('Delete')
    if load_button:
        if selected_file.endswith('.csv'):
            st.session_state.df = pd.read_csv(selected_file)
        elif selected_file.endswith('.tsv'):
            st.session_state.df = pd.read_csv(selected_file, sep='\t')
        elif selected_file.endswith('.xlsx'):
            st.session_state.df = pd.read_excel(selected_file)
    
    if delete_button:
        os.remove(selected_file)
        st.sidebar.write(f"File '{selected_file}' has been deleted.")
        st.experimental_rerun()
    
    if refresh_button:
        st.session_state.df = []

st.sidebar.markdown('---')

st.sidebar.subheader('Save Updated DataFrame')
save_title = st.sidebar.text_input('Enter file name', 'updated_dataframe')
save_button = st.sidebar.button('Save')

if save_button:
    today_dir = os.path.join('appfile/data', datetime.now().strftime("%Y-%m-%d"))
    if not os.path.exists(today_dir):
        os.makedirs(today_dir)

    updated_file_path = os.path.join(today_dir, f'{save_title}.csv')
    st.session_state.df.to_csv(updated_file_path, index=False)
    st.sidebar.write(f"Updated file saved to {updated_file_path}")

st.sidebar.markdown('---')

#if st.session_state.df is not None:

visualization = DataVisualization(st.session_state.df)
visualization.show_columns_info()
visualization.show_numeric_describe()
visualization.show_non_numeric_describe()
visualization.show_top_10_rows()
visualization.show_missing_ratio()

preprocessing = Preprocessing(st.session_state.df)
preprocessing.remove_columns()
preprocessing.fill_missing_values()
preprocessing.apply_encoding()
preprocessing.apply_scaling()
preprocessing.create_new_features()
"""