import streamlit as st
import numpy as np
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler, RandomSampler
from optuna.visualization import plot_param_importances, plot_slice
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

class LightGBMTuner:
    def __init__(self):
        self.params_default = {
            'objective': 'mae',
            'random_seed': 1234,
            'learning_rate': 0.02,
            'min_data_in_bin': 3,
            'bagging_freq': 1,
            'bagging_seed': 0,
            'verbose': -1,
        }

    def streamlit_ui(self):
        # Streamlitインターフェースの作成
        st.subheader("LightGBM Hyperparameter Presetting")

        # ハイパーパラメータの入力
        objective = st.selectbox('Objective', ['mae', 'mse', 'binary'], index=0)
        random_seed = st.number_input('Random Seed', value=self.params_default['random_seed'])
        learning_rate = st.number_input('Learning Rate', value=self.params_default['learning_rate'])
        min_data_in_bin = st.number_input('Min Data in Bin', value=self.params_default['min_data_in_bin'])
        bagging_freq = st.number_input('Bagging Frequency', value=self.params_default['bagging_freq'])
        bagging_seed = st.number_input('Bagging Seed', value=self.params_default['bagging_seed'])
        verbose = st.selectbox('Verbose', [-1, 0, 1], index=0)

        # Samplerの選択
        sampler_option = st.radio('Sampler', ['TPESampler', 'RandomSampler'], index=0)

        # 選択されたハイパーパラメータを設定
        self.params_base = {
            'objective': objective,
            'random_seed': random_seed,
            'learning_rate': learning_rate,
            'min_data_in_bin': min_data_in_bin,
            'bagging_freq': bagging_freq,
            'bagging_seed': bagging_seed,
            'verbose': verbose,
        }

        if sampler_option == 'TPESampler':
            self.sampler = TPESampler(seed=0)
        else:
            self.sampler = RandomSampler(seed=0)

    def objective(self, trial):
        params_tuning = {
            'num_leaves': trial.suggest_int('num_leaves', 50, 200),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 2, 30),
            'max_bin': trial.suggest_int('max_bin', 200, 400),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 0.95),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.35, 0.65),
            'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.01, 1, log=True),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.01, 1, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 0.01, 1, log=True)
        }

        # 探索用ハイパーパラメータの設定
        params_tuning.update(self.params_base)
        lgb_train = lgb.Dataset(self.x_tr, label=self.y_tr)
        lgb_eval = lgb.Dataset(self.x_va, label=self.y_va, reference=lgb_train)

        # 探索用ハイパーパラメータで学習
        model = lgb.train(params_tuning,
                          lgb_train,
                          num_boost_round=10000,
                          valid_sets=[lgb_train, lgb_eval],
                          valid_names=['train', 'valid'],
                          callbacks=[
                              lgb.early_stopping(100),
                              lgb.log_evaluation(500)
                          ])
        y_va_pred = model.predict(self.x_va, num_iteration=model.best_iteration)
        score = mean_absolute_error(self.y_va, y_va_pred)
        return score

    def run_optimization(self, x_tr, y_tr, x_va, y_va):
        self.x_tr = x_tr
        self.y_tr = y_tr
        self.x_va = x_va
        self.y_va = y_va

        study = optuna.create_study(sampler=self.sampler, direction='minimize')
        study.optimize(self.objective, n_trials=200)

        # 最適化の結果を確認
        trial = study.best_trial
        st.write(f'trial {trial.number}')
        st.write(f'MAE best: {trial.value:.2f}')
        st.write('Best Hyperparameters:', trial.params)

        # 最適化ハイパーパラメータの設定
        params_best = trial.params
        params_best.update(self.params_base)
        st.write('Best Parameters:', params_best)

        # ハイパーパラメータの重要度の可視化
        fig = plot_param_importances(study)
        st.plotly_chart(fig)

'''
def main():
    tuner = LightGBMTuner()
    tuner.streamlit_ui()
    if st.button("Optimize"):
        # preprocessing.py からデータを取得
        required_keys = ['x_tr', 'y_tr', 'x_va', 'y_va']
        if all(key in st.session_state for key in required_keys):
            tuner.run_optimization(
                st.session_state.x_tr,
                st.session_state.y_tr,
                st.session_state.x_va,
                st.session_state.y_va
            )
        else:
            st.error("Data not found in session state. Please run data preprocessing first.")

if __name__ == "__main__":
    main()
'''