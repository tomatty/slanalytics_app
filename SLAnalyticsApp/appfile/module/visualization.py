import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import io

class DataVisualization:
    def __init__(self, df):
        self.df = df

    def show_columns_info(self):
        if st.sidebar.checkbox('Show columns info', value=True):
            buffer = io.StringIO()
            self.df.info(buf=buffer)
            s = buffer.getvalue()
            st.text("Columns information:")
            st.text(s)

    def show_numeric_describe(self):
        if st.sidebar.checkbox('Show numeric describe', value=True):
            st.write("Basic numeric statistics:")
            st.write(self.df.describe())

    def show_non_numeric_describe(self):
        if st.sidebar.checkbox('Show non-numeric describe', value=True):
            st.write("Basic non-numeric statistics:")
            st.write(self.df.describe(include='O'))

    def show_top_100_rows(self):
        if st.sidebar.checkbox('Show top 100 rows', value=True):
            st.write("Dataframe preview:")
            st.dataframe(self.df.head(100))

    def show_missing_ratio(self):
        if st.sidebar.checkbox('Show missing ratio', value=True):
            st.write("Missing Ratio:")
            df_na = (self.df.isnull().sum() / len(self.df)) * 100
            df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:30]
            missing_data = pd.DataFrame({'Missing Ratio': df_na})
            st.dataframe(missing_data.head(50))

    def show_histogram(self):
        if st.sidebar.checkbox('Show histogram'):
            numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numeric_columns) > 0:
                selected_columns = st.multiselect('Select columns for histogram', numeric_columns)
                nbins = st.slider('Select number of bins', min_value=5, max_value=100, value=30)
                for column in selected_columns:
                    fig = px.histogram(self.df, x=column, nbins=nbins, title=f'Histogram of {column}')
                    st.plotly_chart(fig)

    def show_correlation_heatmap(self):
        if st.sidebar.checkbox('Show correlation heatmap'):
            corr_matrix = self.df.corr()
            fig = ff.create_annotated_heatmap(
                z=corr_matrix.values,
                x=list(corr_matrix.columns),
                y=list(corr_matrix.index),
                colorscale='Viridis',
                showscale=True,
                annotation_text=corr_matrix.round(2).values
            )
            fig.update_layout(
                width=1600,
                height=800
            )
            st.plotly_chart(fig, use_container_width=True)

    def show_pairplot(self):
        if st.sidebar.checkbox('Show pairplot'):
            numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            all_columns = ['Select all'] + numeric_columns
            selected_columns = st.multiselect('Select columns for pairplot', all_columns, default=['Select all'])
            if 'Select all' in selected_columns:
                selected_columns = numeric_columns
            if selected_columns:
                pairplot_fig = sns.pairplot(self.df[selected_columns])
                st.pyplot(pairplot_fig)

    def show_box_plot(self):
        if st.sidebar.checkbox('Show box plot'):
            numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numeric_columns) > 0:
                all_columns = ['Select all'] + numeric_columns
                selected_columns = st.multiselect('Select columns for box plot', all_columns)
                if 'Select all' in selected_columns:
                    selected_columns = numeric_columns
                if selected_columns:
                    fig = px.box(self.df, y=selected_columns, title=f'Box Plot of {", ".join(selected_columns)}')
                    st.plotly_chart(fig)

    def show_pareto_chart(self):
        if st.sidebar.checkbox('Show Pareto Chart'):
            all_columns = self.df.columns.tolist()
            selected_column = st.selectbox('Select column for Pareto Chart', all_columns)
            if selected_column:
                if self.df[selected_column].dtype in ['float64', 'int64']:
                    num_bins = st.slider('Select number of bins', min_value=5, max_value=50, value=10)
                    binned_data = pd.cut(self.df[selected_column], bins=num_bins)
                    data = binned_data.value_counts().sort_values(ascending=False)
                    data.index = data.index.astype(str)  # インターバルオブジェクトを文字列に変換
                    cumulative_percentage = data.cumsum() / data.sum() * 100
                    x_data = data.index
                else:
                    data = self.df[selected_column].value_counts().sort_values(ascending=False)
                    cumulative_percentage = data.cumsum() / data.sum() * 100
                    x_data = data.index
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=x_data,
                    y=data.values,
                    name='Count'
                ))
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=cumulative_percentage,
                    name='Cumulative %',
                    yaxis='y2',
                    mode='lines+markers'
                ))
                fig.update_layout(
                    title=f'Pareto Chart of {selected_column}',
                    xaxis_title=selected_column if self.df[selected_column].dtype not in ['float64', 'int64'] else f'{selected_column} Bins',
                    yaxis_title='Count',
                    yaxis2=dict(
                        title='Cumulative %',
                        overlaying='y',
                        side='right',
                        range=[0, 100]
                    )
                )
                st.plotly_chart(fig)

    def show_pivot_table(self):
        if st.sidebar.checkbox('Show Pivot Table'):
            category_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            numeric_columns = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            st.subheader('Filter Data')
            selected_filters = st.multiselect('Select columns to filter by', category_columns + numeric_columns)
            filter_conditions = {}
            for col in selected_filters:
                if col in category_columns:
                    options = st.multiselect(f'Filter values for {col}', self.df[col].unique())
                    filter_conditions[col] = options
                else:
                    min_val, max_val = st.slider(f'Select range for {col}', 
                                                 float(self.df[col].min()), 
                                                 float(self.df[col].max()), 
                                                 (float(self.df[col].min()), float(self.df[col].max())))
                    filter_conditions[col] = (min_val, max_val)
            filtered_df = self.df.copy()
            for col, condition in filter_conditions.items():
                if col in category_columns:
                    if condition:
                        filtered_df = filtered_df[filtered_df[col].isin(condition)]
                else:
                    filtered_df = filtered_df[(filtered_df[col] >= condition[0]) & (filtered_df[col] <= condition[1])]
            st.subheader('Select columns for pivot table')
            index_columns = st.multiselect('Select index columns', category_columns, key='index')
            column_columns = st.multiselect('Select columns for pivot columns', category_columns, key='columns')
            value_column = st.selectbox('Select value column', numeric_columns, key='values')
            aggfunc = st.selectbox('Select aggregation function', ['sum', 'mean', 'count', 'min', 'max', 'std', 'median'])
            if index_columns and column_columns and value_column:
                pivot_table = filtered_df.pivot_table(
                    index=index_columns,
                    columns=column_columns,
                    values=value_column,
                    aggfunc=aggfunc
                )
                st.write('Pivot Table')
                st.dataframe(pivot_table)

    def show_violin_plot(self):
        if st.sidebar.checkbox('Show violin plot'):
            columns = self.df.columns.tolist()
            selected_columns = st.multiselect('Select columns for violin plot', columns)
            target_column = st.selectbox('Select target column for violin plot', columns)
            if selected_columns and target_column:
                fig = px.violin(self.df, y=selected_columns, x=target_column, title=f'Violin Plot of {", ".join(selected_columns)} against {target_column}')
                st.plotly_chart(fig)

    def show_value_counts(self):
        if st.sidebar.checkbox('Show value counts'):
            columns = self.df.columns.tolist()
            selected_column = st.selectbox('Select column for value counts', columns)
            if selected_column:
                value_counts = self.df[selected_column].value_counts()
                st.write(f'Value counts for {selected_column}:')
                st.write(value_counts)
