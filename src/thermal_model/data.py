import pandas as pd
import numpy as np
import os
from keys import *
from utils_thermal_model_raw_process import *
import math
from tqdm import tqdm
from loader import Loader


def standard_data_prepare(df, history_ambt_temp):
    """对于原始数据进行标准化的操作，方便对单独的日期进行建模"""
    df = rename_excel_raw_data_from_electrolyzer(df)  # 对原始的列名进行重命名
    df = trim_abnormal_raw_data(df)  # 删除存在-9999的异常数值
    df = fillin_absent_data(df)  # 如果原始数据存在缺失，则进行填补
    df = differential_temperature_raw_data(df)  # 对温度进行小波变换，并且进行差分处理
    df = cell_voltage_current_density(df)  # 计算电流密度与小室电压
    df = voltage_thermal_neutral(df)  # 根据出口温度计算热中性电压，可以用来计算发热，热中性电压高于测量值的时候，还得看一下怎么处理
    df = convert_df_time_raw_data(df)  # 对原始数据中的中文时间进行转换
    df = fill_history_ambt_temperature(df, history_ambt_temp)  # 给历史数据中添加环境温度
    return df


class ThermalModelData(Loader):
    """自动判断是否已经进行了excel的转化，如果没有就先转换excel然后再存成pickle"""

    def __init__(
        self, 
        mode = 'all'
        ) -> None:
        if mode == 'all':
            fp_cache=Cache.thermal_model_data_1101
        else:
            fp_cache=Cache.thermal_model_data_0102

        super().__init__(fp_cache)
        self.mode = mode
        self.source_folder = DataDir.Int_thermal_model
        
        if not os.path.exists(DataDir.Int_thermal_model):
            os.makedirs(
                DataDir.Int_thermal_model,
            )

    @staticmethod
    def read_convert_raw_data_from_excel():
        print("---Reading raw data from excel and preprocessing")
        source_folder = os.path.join(
            DataDir.Raw_electrolyzer, os.listdir(DataDir.Raw_electrolyzer)[0]
        )  # 只读取2021年的实验部分

        history_ambt_temp = pd.read_csv(Files.history_ambient_temperature)
        for file in tqdm(os.listdir(source_folder)):  # 只选取第一部分的数据
            df = pd.read_excel(os.path.join(source_folder, file))
            df = standard_data_prepare(df, history_ambt_temp)
            df.to_csv(os.path.join(DataDir.Int_thermal_model, file), encoding=ENCODING)

    def concat_all_raw_data(self):
        raw_data_list = []
        print("---Concatenating all raw data from csv")
        for file in os.listdir(self.source_folder):
            df_cur = pd.read_csv(os.path.join(self.source_folder, file))
            raw_data_list.append(df_cur)
        df_raw_data = pd.concat(raw_data_list, ignore_index=True)
        return df_raw_data
    
    def concat_pseudo_static_data(self):
        raw_data_list = []
        file_list = [
            '1001',
            '1014',
            '1007',
            '1029',
            '1123',
            '1128',
            '0924'
        ]
        print("---Concatenating pseudo static data from csv")
        for file in os.listdir(self.source_folder):
            if file[7:11] in file_list:
                df_cur = pd.read_csv(os.path.join(self.source_folder, file))
                raw_data_list.append(df_cur)
        df_raw_data = pd.concat(raw_data_list, ignore_index=True)
        return df_raw_data

    def run(self):
        if len(os.listdir(self.source_folder)) < 1:
            ThermalModelData.read_convert_raw_data_from_excel()
        if self.mode == 'all':
            df_raw_data = self.concat_all_raw_data()
        else:
            df_raw_data = self.concat_pseudo_static_data()

        return df_raw_data


def generate_model_input(df):
    """在已有的数据基础上，生成线性模型所需的输入列"""
    df[Cols.electric_heat] = np.maximum(
        0,
        (
            df[Cols.stack_voltage]
            - df[Cols.voltage_thermal_neutral] * Constants.num_cells
        )
        * df[Cols.stack_current]
        / 1000,
    )  # 生成电热，如果采集到的电压信号低于理论热中性电压，则取零

    df[Cols.radiation_dissipation] = (
        (df[Cols.temp_out] + Constants.absolute_temp_delta) ** 4 / 2
        + (df[Cols.lye_temp] + Constants.absolute_temp_delta) ** 4 / 2
        - (df[Cols.ambt_temp] + Constants.absolute_temp_delta) ** 4
    )  # 生成辐射散热，将出入口温度做平均考虑

    df[Cols.input_lye_heat] = df[Cols.lye_flow] * df[Cols.lye_temp]  # 跟随碱液带入的热量

    df[Cols.output_lye_heat] = df[Cols.lye_flow] * df[Cols.temp_out]

    return df
