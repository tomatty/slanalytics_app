import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
from datetime import datetime
import time
import os
import io
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'module')))

from visualization import DataVisualization

st.sidebar.markdown('---')

# Display a title
st.title('Exploratory Data Analysis')

# ファイルアップローダー
st.sidebar.subheader('Upload train and test files')
uploaded_train_file = st.sidebar.file_uploader("Choose a train file", type=['csv', 'tsv', 'xlsx'], key='train')
uploaded_test_file = st.sidebar.file_uploader("Choose a test file", type=['csv', 'tsv', 'xlsx'], key='test')

def check_file(file):
    # ファイルサイズチェック（5MB以下）
    if file.size > 5 * 1024 * 1024:
        return False, "File size should be less than 5MB"
    
    # ファイル拡張子チェック
    if not file.name.endswith(('.csv', '.tsv', '.xlsx')):
        return False, "Unsupported file type"
    
    return True, None

train_data = None
test_data = None

# セッションステートにdfを初期化
if 'df' not in st.session_state:
    st.session_state.df = None

# セッションステートにデータフレーム作成フラグを初期化
if 'create_dataframe' not in st.session_state:
    st.session_state.create_dataframe = False

if uploaded_train_file and uploaded_test_file:
    # ファイルチェック
    is_valid_train, error_message_train = check_file(uploaded_train_file)
    is_valid_test, error_message_test = check_file(uploaded_test_file)
    
    if is_valid_train and is_valid_test:
        # ファイルをデータフレームに読み込む
        try:
            if uploaded_train_file.name.endswith('.csv'):
                train_data = pd.read_csv(uploaded_train_file)
            elif uploaded_train_file.name.endswith('.tsv'):
                train_data = pd.read_csv(uploaded_train_file, sep='\t')
            elif uploaded_train_file.name.endswith('.xlsx'):
                train_data = pd.read_excel(uploaded_train_file)

            if uploaded_test_file.name.endswith('.csv'):
                test_data = pd.read_csv(uploaded_test_file)
            elif uploaded_test_file.name.endswith('.tsv'):
                test_data = pd.read_csv(uploaded_test_file, sep='\t')
            elif uploaded_test_file.name.endswith('.xlsx'):
                test_data = pd.read_excel(uploaded_test_file)
            
            st.sidebar.write(f"Train file '{uploaded_train_file.name}' and Test file '{uploaded_test_file.name}' are ready.")
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                confirm_button = st.button('Create DataFrame')
            with col2:
                refresh_button = st.button('Refresh')
            
            if confirm_button:
                if train_data is not None and test_data is not None:
                    st.session_state.df = pd.concat([train_data, test_data], ignore_index=True)
                    st.sidebar.success("Files successfully combined!")
                    st.session_state.create_dataframe = True
                else:
                    st.sidebar.error("Failed to read the uploaded files.")
            if refresh_button:
                st.session_state.df = None
                st.session_state.create_dataframe = False
                st.experimental_rerun()
                
        except Exception as e:
            st.sidebar.error(f"Error processing files: {e}")
    else:
        if not is_valid_train:
            st.sidebar.write(f"Error in train file: {error_message_train}")
        if not is_valid_test:
            st.sidebar.write(f"Error in test file: {error_message_test}")

st.sidebar.markdown('---')

if st.session_state.create_dataframe and st.session_state.df is not None:
    visualization = DataVisualization(st.session_state.df)
    visualization.show_top_100_rows()
    visualization.show_columns_info()
    visualization.show_numeric_describe()
    visualization.show_non_numeric_describe()
    visualization.show_missing_ratio()
    visualization.show_correlation_heatmap()
    visualization.show_pairplot()
    visualization.show_histogram()
    visualization.show_box_plot()
    visualization.show_violin_plot()
    visualization.show_pareto_chart()
    visualization.show_pivot_table()


"""
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

# ファイルアップローダー
st.sidebar.subheader('Upload a file')
uploaded_file = st.sidebar.file_uploader("Choose a file", type=['csv', 'tsv', 'xlsx'])

def check_file(file):
    # ファイルサイズチェック（5MB以下）
    if file.size > 10 * 1024 * 1024 *1024:
        return False, "File size should be less than 5MB"
    
    # ファイル拡張子チェック
    if not file.name.endswith(('.csv', '.tsv', '.xlsx')):
        return False, "Unsupported file type"
    
    return True, None

if uploaded_file is not None:
    # ファイルチェック
    is_valid, error_message = check_file(uploaded_file)
    
    if is_valid:
        # 一時的にファイルを保存せず、確認ボタンを表示
        st.sidebar.write(f"File '{uploaded_file.name}' is ready for upload.")
        confirm_button = st.sidebar.button('Confirm Upload')
        
        if confirm_button:
            # ファイルの保存
            file_path = os.path.join(today_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.sidebar.write(f"File '{uploaded_file.name}' uploaded successfully to {today_dir}!")
    else:
        st.sidebar.write(f"Error: {error_message}")

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
"""



