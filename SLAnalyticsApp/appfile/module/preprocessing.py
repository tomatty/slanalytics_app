import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer, LabelEncoder
import category_encoders as ce
from sklearn.model_selection import train_test_split, KFold


class Preprocessing:
    def __init__(self, df):
        self.df = df

    def remove_columns(self):
        if st.sidebar.checkbox('Remove Columns'):
            st.subheader('Remove Columns')
            columns_to_remove = st.multiselect('Select columns to remove', self.df.columns.tolist())
            if st.button('Remove'):
                self.df.drop(columns=columns_to_remove, inplace=True)
                st.write('Updated DataFrame')
                st.dataframe(self.df)

    def fill_missing_values(self):
        if st.sidebar.checkbox('Fill Missing Values'):
            st.subheader('Fill Missing Values')
            
            fill_options = st.radio(
                "Select fill method",
                ['Simple Fill', 'Group by Categorical Variables', 'Group by Binned Numerical Variables']
            )
            
            column_to_fill = st.selectbox('Select column to fill', self.df.columns.tolist())
            fill_method = st.selectbox('Select fill value method', ['Mean', 'Median', 'Mode', 'Previous Value', 'Next Value', 'Custom Value'])

            if fill_method == 'Custom Value':
                custom_value = st.text_input('Enter custom value')

            if fill_options == 'Group by Categorical Variables':
                categorical_columns = st.multiselect('Select categorical columns to group by', self.df.select_dtypes(include=['object', 'category']).columns.tolist())

            if fill_options == 'Group by Binned Numerical Variables':
                numerical_columns = st.multiselect('Select numerical columns to bin and group by', self.df.select_dtypes(include=['float64', 'int64']).columns.tolist())
                bin_size = st.number_input('Select number of bins', min_value=1, max_value=100, value=3)
            
            if st.button('Fill'):
                if fill_options == 'Simple Fill':
                    if fill_method == 'Mean':
                        self.df[column_to_fill].fillna(self.df[column_to_fill].mean(), inplace=True)
                    elif fill_method == 'Median':
                        self.df[column_to_fill].fillna(self.df[column_to_fill].median(), inplace=True)
                    elif fill_method == 'Mode':
                        self.df[column_to_fill].fillna(self.df[column_to_fill].mode()[0], inplace=True)
                    elif fill_method == 'Previous Value':
                        self.df[column_to_fill].fillna(method='ffill', inplace=True)
                    elif fill_method == 'Next Value':
                        self.df[column_to_fill].fillna(method='bfill', inplace=True)
                    elif fill_method == 'Custom Value':
                        self.df[column_to_fill].fillna(custom_value, inplace=True)

                elif fill_options == 'Group by Categorical Variables' and categorical_columns:
                    if fill_method == 'Mean':
                        self.df[column_to_fill] = self.df.groupby(categorical_columns)[column_to_fill].transform(lambda x: x.fillna(x.mean()))
                    elif fill_method == 'Median':
                        self.df[column_to_fill] = self.df.groupby(categorical_columns)[column_to_fill].transform(lambda x: x.fillna(x.median()))
                    elif fill_method == 'Mode':
                        self.df[column_to_fill] = self.df.groupby(categorical_columns)[column_to_fill].transform(lambda x: x.fillna(x.mode()[0]))
                    elif fill_method == 'Previous Value':
                        self.df[column_to_fill] = self.df.groupby(categorical_columns)[column_to_fill].transform(lambda x: x.fillna(method='ffill'))
                    elif fill_method == 'Next Value':
                        self.df[column_to_fill] = self.df.groupby(categorical_columns)[column_to_fill].transform(lambda x: x.fillna(method='bfill'))
                    elif fill_method == 'Custom Value':
                        self.df[column_to_fill] = self.df.groupby(categorical_columns)[column_to_fill].transform(lambda x: x.fillna(custom_value))
                
                elif fill_options == 'Group by Binned Numerical Variables' and numerical_columns:
                    for col in numerical_columns:
                        self.df[f'{col}_bin'] = pd.cut(self.df[col], bins=bin_size)
                    group_columns = [f'{col}_bin' for col in numerical_columns]
                    if fill_method == 'Mean':
                        self.df[column_to_fill] = self.df.groupby(group_columns)[column_to_fill].transform(lambda x: x.fillna(x.mean()))
                    elif fill_method == 'Median':
                        self.df[column_to_fill] = self.df.groupby(group_columns)[column_to_fill].transform(lambda x: x.fillna(x.median()))
                    elif fill_method == 'Mode':
                        self.df[column_to_fill] = self.df.groupby(group_columns)[column_to_fill].transform(lambda x: x.fillna(x.mode()[0]))
                    elif fill_method == 'Previous Value':
                        self.df[column_to_fill] = self.df.groupby(group_columns)[column_to_fill].transform(lambda x: x.fillna(method='ffill'))
                    elif fill_method == 'Next Value':
                        self.df[column_to_fill] = self.df.groupby(group_columns)[column_to_fill].transform(lambda x: x.fillna(method='bfill'))
                    elif fill_method == 'Custom Value':
                        self.df[column_to_fill] = self.df.groupby(group_columns)[column_to_fill].transform(lambda x: x.fillna(custom_value))
                
                st.write(f"Missing values in column '{column_to_fill}' filled using method '{fill_method}'")
                st.dataframe(self.df)

    def apply_encoding(self):
        if st.sidebar.checkbox('Use Encoding'):
            st.subheader('Encoding Options')
            
            encoding_type = st.radio(
                "Select encoding type",
                ['One-Hot Encoding', 'Label Encoding', 'Target Encoding']
            )
            
            columns_to_encode = st.multiselect('Select columns to encode', self.df.columns.tolist())

            target_column = None
            if encoding_type == 'Target Encoding':
                target_column = st.selectbox('Select target column for target encoding', self.df.columns.tolist())
            
            if st.button('Apply Encoding'):
                if encoding_type == 'One-Hot Encoding':
                    self.df = pd.get_dummies(self.df, columns=columns_to_encode)
                elif encoding_type == 'Label Encoding':
                    label_encoder = LabelEncoder()
                    for column in columns_to_encode:
                        self.df[column] = label_encoder.fit_transform(self.df[column])
                elif encoding_type == 'Target Encoding' and target_column:
                    target_encoder = ce.TargetEncoder(cols=columns_to_encode, smoothing=0.5)
                    self.df[columns_to_encode] = target_encoder.fit_transform(self.df[columns_to_encode], self.df[target_column])
                
                st.write(f"Applied {encoding_type} on columns '{', '.join(columns_to_encode)}'")
                st.dataframe(self.df)

    def apply_scaling(self):
        if st.sidebar.checkbox('Use Scaling'):
            st.subheader('Scaling Options')

            columns_to_scale = st.multiselect('Select columns to scale', self.df.select_dtypes(include=['float64', 'int64']).columns.tolist())
            scaling_method = st.radio(
                "Select scaling method",
                ['Standardization', 'Min-Max Scaling', 'Log(x+1)', 'Box-Cox', 'Clipping', 'Binning', 'RankGauss']
            )

            clip_min = None
            clip_max = None
            bin_size = None
            if scaling_method == 'Clipping':
                clip_min = st.number_input('Enter minimum value for clipping', value=float(self.df[columns_to_scale].min().min()))
                clip_max = st.number_input('Enter maximum value for clipping', value=float(self.df[columns_to_scale].max().max()))

            if scaling_method == 'Binning':
                bin_size = st.number_input('Enter number of bins', min_value=1, max_value=100, value=3)

            if st.button('Apply Scaling'):
                if scaling_method == 'Standardization':
                    scaler = StandardScaler()
                    self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])
                elif scaling_method == 'Min-Max Scaling':
                    scaler = MinMaxScaler()
                    self.df[columns_to_scale] = scaler.fit_transform(self.df[columns_to_scale])
                elif scaling_method == 'Log(x+1)':
                    self.df[columns_to_scale] = np.log1p(self.df[columns_to_scale])
                elif scaling_method == 'Box-Cox':
                    pt = PowerTransformer(method='box-cox')
                    self.df[columns_to_scale] = pt.fit_transform(self.df[columns_to_scale])
                elif scaling_method == 'Clipping':
                    self.df[columns_to_scale] = self.df[columns_to_scale].clip(lower=clip_min, upper=clip_max)
                elif scaling_method == 'Binning':
                    for col in columns_to_scale:
                        self.df[f'{col}_binned'] = pd.cut(self.df[col], bins=bin_size)
                elif scaling_method == 'RankGauss':
                    qt = QuantileTransformer(output_distribution='normal')
                    self.df[columns_to_scale] = qt.fit_transform(self.df[columns_to_scale])

                st.write(f"Applied {scaling_method} on columns '{', '.join(columns_to_scale)}'")
                st.dataframe(self.df)

    def create_new_features(self):
        if st.sidebar.checkbox('Create New Features'):
            st.subheader('Create New Feature')

            new_column_name = st.text_input('Enter new column name')

            existing_columns = self.df.columns.tolist()
            selected_columns = st.multiselect('Select columns to use in calculation', existing_columns)

            calculation_formula = st.text_area('Enter calculation formula (use column names as variables)')

            if st.button('Create Feature'):
                try:
                    # 安全に計算式を評価するために、既存の列をローカル変数に設定
                    local_vars = {col: self.df[col] for col in selected_columns}
                    self.df[new_column_name] = eval(calculation_formula, {"np": np}, local_vars)
                    st.write(f"New feature '{new_column_name}' created successfully")
                    st.dataframe(self.df)
                except Exception as e:
                    st.error(f"Error in creating new feature: {e}")


class DataSplitter:
    def __init__(self, df, target_column, split_method, test_size=0.2, second_test_size=0.2, n_splits=5, random_state=42):
        self.df = df
        self.target_column = target_column
        self.split_method = split_method
        self.test_size = test_size
        self.second_test_size = second_test_size
        self.n_splits = n_splits
        self.random_state = random_state

    def split_data(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        if self.split_method == 'Holdout Only':
            x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            return x_train, x_valid, y_train, y_valid

        elif self.split_method == 'Cross-Validation':
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            for train_index, val_index in kf.split(X):
                x_train, x_valid = X.iloc[train_index], X.iloc[val_index]
                y_train, y_valid = y.iloc[train_index], y.iloc[val_index]
                return x_train, x_valid, y_train, y_valid

        elif self.split_method == '1st Holdout, 2nd Holdout':
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            x_tr, x_va, y_tr, y_va = train_test_split(x_train, y_train, test_size=self.second_test_size, random_state=self.random_state)
            return x_tr, x_va, y_tr, y_va, x_test, y_test

        elif self.split_method == '1st Holdout, 2nd Cross-Validation':
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            for train_index, val_index in kf.split(x_train):
                x_tr, x_va = x_train.iloc[train_index], x_train.iloc[val_index]
                y_tr, y_va = y_train.iloc[train_index], y_train.iloc[val_index]
                return x_tr, x_va, y_tr, y_va, x_test, y_test

def display_and_split_data(df):
    st.subheader('Data Splitting Options')
    
    target_column = st.selectbox('Select target column', df.columns.tolist())
    split_method = st.radio('Select split method', ['Holdout Only', 'Cross-Validation', '1st Holdout, 2nd Holdout', '1st Holdout, 2nd Cross-Validation'])

    test_size, second_test_size, n_splits = None, None, None

    if split_method in ['Holdout Only', '1st Holdout, 2nd Holdout', '1st Holdout, 2nd Cross-Validation']:
        test_size = st.slider('Select test size', min_value=0.1, max_value=0.9, value=0.2)
    
    if split_method in ['1st Holdout, 2nd Holdout', '1st Holdout, 2nd Cross-Validation']:
        second_test_size = st.slider('Select test size for 2nd split', min_value=0.1, max_value=0.9, value=0.2)
    
    if split_method in ['Cross-Validation', '1st Holdout, 2nd Cross-Validation']:
        n_splits = st.number_input('Number of folds', min_value=2, max_value=10, value=5)

    random_state = st.number_input('Random state', value=42)

    if st.button('Split Data'):
        splitter = DataSplitter(
            df=df,
            target_column=target_column,
            split_method=split_method,
            test_size=test_size,
            second_test_size=second_test_size,
            n_splits=n_splits,
            random_state=random_state
        )

        if split_method == 'Holdout Only':
            x_train, x_valid, y_train, y_valid = splitter.split_data()
            col1, col2 = st.columns(2)
            with col1:
                st.write('Training data')
                st.dataframe({'x_train_shape':[x_train.shape],
                            'y_train_shape':[y_train.shape]})
            with col2:
                st.write('Validation data')
                st.dataframe({'x_valid_shape':[x_valid.shape],
                            'y_valid_shape':[y_valid.shape]})
            st.session_state.x_train = x_train
            st.session_state.x_valid = x_valid
            st.session_state.y_train = y_train
            st.session_state.y_valid = y_valid

        elif split_method == 'Cross-Validation':
            x_train, x_valid, y_train, y_valid = splitter.split_data()
            col1, col2 = st.columns(2)
            with col1:
                st.write('Training data')
                st.dataframe({'x_train_shape':[x_train.shape],
                            'y_train_shape':[y_train.shape]})
            with col2:
                st.write('Validation data')
                st.dataframe({'x_valid_shape':[x_valid.shape],
                            'y_valid_shape':[y_valid.shape]})
            st.session_state.x_train = x_train
            st.session_state.x_valid = x_valid
            st.session_state.y_train = y_train
            st.session_state.y_valid = y_valid

        elif split_method == '1st Holdout, 2nd Holdout':
            x_tr, x_va, y_tr, y_va, x_test, y_test = splitter.split_data()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write('Training data')
                st.dataframe({'x_train_shape':[x_tr.shape],
                            'y_train_shape':[y_tr.shape]})
            with col2:
                st.write('Validation data')
                st.dataframe({'x_valid_shape':[x_va.shape],
                            'y_valid_shape':[y_va.shape]})
            with col3:
                st.write('Test data')
                st.dataframe({'x_test_shape':[x_test.shape],
                            'y_test_shape':[y_test.shape]})
            st.session_state.x_tr = x_tr
            st.session_state.x_va = x_va
            st.session_state.y_tr = y_tr
            st.session_state.y_va = y_va
            st.session_state.x_test = x_test
            st.session_state.y_test = y_test

        elif split_method == '1st Holdout, 2nd Cross-Validation':
            x_tr, x_va, y_tr, y_va, x_test, y_test = splitter.split_data()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write('Training data')
                st.dataframe({'x_train_shape':[x_tr.shape],
                            'y_train_shape':[y_tr.shape]})
            with col2:
                st.write('Validation data')
                st.dataframe({'x_valid_shape':[x_va.shape],
                            'y_valid_shape':[y_va.shape]})
            with col3:
                st.write('Test data')
                st.dataframe({'x_test_shape':[x_test.shape],
                            'y_test_shape':[y_test.shape]})
            st.session_state.x_tr = x_tr
            st.session_state.x_va = x_va
            st.session_state.y_tr = y_tr
            st.session_state.y_va = y_va
            st.session_state.x_test = x_test
            st.session_state.y_test = y_test