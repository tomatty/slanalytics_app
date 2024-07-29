import streamlit as st
import importlib.util
import os

# 表示名リスト
modules = ['Explanatory Data Analysis(Available)',
           'Feature Engineering(Under Construction)',
           'Modeling(Under Construction)']

# 表示名とファイルパスのマッピング
module_paths = {
    'Explanatory Data Analysis(Available)': 'pages/eda.py',
    'Feature Engineering(Under Construction)': 'pages/feature.py',
    'Modeling(Under Construction)': 'pages/modeling.py'
}

# サイドバーのセレクトボックス
module = st.sidebar.radio('Select Module', modules, index=0)

# 選択されたファイルパスを取得
base_path = os.path.dirname(__file__)
module_path = os.path.join(base_path, 'appfile', module_paths[module])

# ファイルの存在をチェック
if os.path.exists(module_path):
    spec = importlib.util.spec_from_file_location("selected_module", module_path)
    selected_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected_module)
else:
    st.error(f"No such file or directory: '{module_path}'")
