from sklearn.linear_model import LinearRegression,Lars
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import os
from keys import *
from plotter import Plotter
from thermal_model.configs import (
    thermal_model_input_cols,
    thermal_model_output_cols
)
from utils_thermal_model_raw_process import standardize
import warnings
warnings.filterwarnings('ignore')
def fit_random_forest(
    df_thermal_model_data_input,
    n_estimators=5
):
    """_summary_

    Args:
        df_thermal_model_data_input (_type_): Index(['Unnamed: 0', 'date_time', 'stack_voltage', 'stack_current', '产氢量',
       '产氢累计量', 'lye_flow', 'temp_in', 'sys_pressure', 'temp_o', 'temp_h',
       '氧侧液位', '氢侧液位', 'OTH', 'HTO', '脱氧上温', '脱氧下温', 'B塔上温', 'B塔下温', 'C塔上温',
       'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力', '进罐温度', '进罐压力', 'temp_out',
       'delta_temp', 'cell_voltage', 'current_density',
       'voltage_thermal_neutral', 'ambt_temp', 'electric_heat',
       'radiation_dissipation', 'input_lye_heat', 'output_lye_heat'],
      dtype='object')
    """
    # df_cur = standardize(df_thermal_model_data_input,thermal_model_input_cols) # 进行归一化的意义不大
    df_cur = df_thermal_model_data_input
    model_input = df_cur[thermal_model_input_cols]
    model_target = df_cur[thermal_model_output_cols]
    model = RandomForestRegressor(
        n_estimators=n_estimators,
    )
    model.fit(
        X = model_input,
        y = model_target,
    )
    score = model.score(model_input,model_target)
    print('---Model score of random forest with {} estimator is: {}'.format(
        n_estimators,score
    ))
    return model,model_input,model_target,score

def fit_LARS(
    df_thermal_model_data_input,
):

    """_summary_

    Args:
        df_thermal_model_data_input (_type_): Index(['Unnamed: 0', 'date_time', 'stack_voltage', 'stack_current', '产氢量',
       '产氢累计量', 'lye_flow', 'temp_in', 'sys_pressure', 'temp_o', 'temp_h',
       '氧侧液位', '氢侧液位', 'OTH', 'HTO', '脱氧上温', '脱氧下温', 'B塔上温', 'B塔下温', 'C塔上温',
       'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力', '进罐温度', '进罐压力', 'temp_out',
       'delta_temp', 'cell_voltage', 'current_density',
       'voltage_thermal_neutral', 'ambt_temp', 'electric_heat',
       'radiation_dissipation', 'input_lye_heat', 'output_lye_heat'],
      dtype='object')
    """
    model_input = df_thermal_model_data_input[thermal_model_input_cols]
    model_target = df_thermal_model_data_input[thermal_model_output_cols]
    model = Lars()
    model.fit(
        X = model_input,
        y = model_target,
    )
    score = model.score(model_input,model_target)
    print('---Model score of Lars is: {}'.format(
        score
    ))
    return model,model_input,model_target

def model_estimator(
    model,
    model_input,
    model_target
):
    model_target = np.squeeze(np.array(model_target))
    model_predict = np.squeeze(
        model.predict(model_input)
    )
    
    error = model_target - model_predict
    return  model_predict, error

