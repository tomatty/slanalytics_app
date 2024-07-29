import streamlit as st
import importlib.util
import os


# 表示名リスト
modules = ['Explanatory Data Analysis',
           'Feature Engineering',
           'Modeling']

# 表示名とファイルパスのマッピング
module_paths = {
    'Explanatory Data Analysis': 'pages/eda.py',
    'Feature Engineering': 'pages/feature.py',
    'Modeling': 'pages/modeling.py'
}

# サイドバーのセレクトボックス
module = st.sidebar.radio('Select Module', modules, index=0)

# 選択されたファイルパスを取得
module_path = os.path.join('appfile', module_paths[module])

# 選択されたファイルをインポートして実行
if module_path:
    spec = importlib.util.spec_from_file_location("selected_module", module_path)
    selected_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected_module)
