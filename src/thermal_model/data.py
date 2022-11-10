import pandas as pd
import numpy as np
import os
from keys import *
from utils_thermal_model_raw_process import *
import math
from tqdm import tqdm
from loader import Loader


class ThermalModelData(Loader):
    """自动判断是否已经进行了excel的转化，如果没有就先转换excel然后再存成pickle"""
    def __init__(
        self,
        fp_cache = Cache.thermal_model_data_1101
    ) -> None:
        super().__init__(fp_cache)

    @staticmethod
    def read_convert_raw_data_from_excel():
        print('---Reading raw data from excel and preprocessing')
        source_folder = os.path.join(
            DataDir.Raw_electrolyzer,
            os.listdir(DataDir.Raw_electrolyzer)[0]
        ) # 只读取2021年的实验部分

        history_ambt_temp = pd.read_csv(
            Files.history_ambient_temperature
        )
        for file in tqdm( 
            os.listdir(source_folder)
        ): # 只选取第一部分的数据
            df = pd.read_excel(
                os.path.join(
                    source_folder,
                    file
                )
            )
            df = rename_excel_raw_data_from_electrolyzer(df)
            df = trim_abnormal_raw_data(df)
            df = fillin_absent_data(df)
            df = differential_temperature_raw_data(df)
            df = cell_voltage_current_density(df)
            df = convert_df_time_raw_data(df)
            df = fill_history_ambt_temperature(
                df,
                history_ambt_temp
            )
            df.to_csv(
                os.path.join(
                    DataDir.Raw_thermal_model,
                    file
                ),
                encoding=ENCODING
            )
    
    def concat_all_raw_data(self):
        source_folder = DataDir.Raw_thermal_model
        if len(os.listdir(source_folder))<1:
            ThermalModelData.read_convert_raw_data_from_excel()
        raw_data_list = []
        print('---Concating all raw data from csv')
        for file in os.listdir(source_folder):
            df_cur = pd.read_csv(
                os.path.join(
                    source_folder,
                    file
                )
            )
            raw_data_list.append(df_cur)
        df_raw_data = pd.concat(
            raw_data_list,
            ignore_index=True
        )
        return df_raw_data


    def run(self):
        df_raw_data = self.concat_all_raw_data()

        return df_raw_data
        