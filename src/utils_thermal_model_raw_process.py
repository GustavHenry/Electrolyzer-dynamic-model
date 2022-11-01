"""

主要针对thermal model进行原始excel的格式化操作使用的函数

"""

import pandas as pd
import numpy as np
from keys import Cols, Constants,TimeForms
import math


def rename_excel_raw_data_from_electrolyzer(excel):
    """对于原始数据excel中的列名进行初步的修改，以符合文件的规范"""
    df_rename = excel.copy()
    df_rename = df_rename.rename(
        columns={
            "时间": Cols.date_time,
            "电解电压": Cols.stack_voltage,
            "电解电流": Cols.stack_current,
            "碱液流量": Cols.lye_flow,
            "碱温": Cols.lye_temp,
            "系统压力  ": Cols.sys_pressure,
            "氧槽温": Cols.o_temp,
            "氢槽温": Cols.h_temp,
            "氧中氢": Cols.o_in_h,
            "氢中氧": Cols.h_in_o,
        }
    )
    return df_rename


def differential_temperature_raw_data(df):
    """这里针对的是计算温度的差分，方法是用当前时刻的温度减去上一时刻的温度"""
    df[Cols.temp_out] = (df[Cols.o_temp] + df[Cols.h_temp]) / 2
    diff_temp = [0]
    for idx in range(1, len(df)):
        diff_temp.append(df.iloc[idx][Cols.temp_out] - df.iloc[idx - 1][Cols.temp_out])
    df[Cols.delta_temp] = diff_temp
    return df

def trim_abnormal_raw_data(df):
    """因为数据中可能存在没有记录上结果，可能会导致数据最有有-9999的部分，所以这里需要进行截断"""
    if min(df[Cols.stack_voltage]) <0:
        for idx in range(len(df)):
            if df.iloc[idx][Cols.stack_voltage]<0:
                df = df.iloc[:idx]
                return df
    else:
        return df

def cell_voltage_current_density(df):
    """计算小室电压与电流密度，以方便未来计算"""
    df[Cols.cell_voltage] = df[Cols.stack_voltage] / Constants.num_cells
    df[Cols.current_density] = df[Cols.stack_current] / Constants.active_area
    return df


def convert_df_time_raw_data(df):
    """将原始表格中的中文年月日等信息，转化成标准的pandas时间格式，并且储存"""

    def convert_time_raw_data(row):

        time_str = row[Cols.date_time]
        time_str = (
            time_str.replace("年", "-")
            .replace("月", "-")
            .replace("日", " ")
            .replace("时", ":")
            .replace("分", ":")
            .replace("秒", "")
        )
        time_stamp = pd.to_datetime(time_str)
        return time_stamp

    df[Cols.date_time] = df.apply(lambda row: convert_time_raw_data(row), axis=1)
    return df


def fill_history_ambt_temperature(df, history_ambt_temp):
    def fill_ambt(row, history_ambt_temp):

        Day_0_min = history_ambt_temp.loc[
            history_ambt_temp[Cols.date] == row[Cols.date_time].strftime(TimeForms.date)
        ].iloc[0]["min_temp"]
        Day_0_max = history_ambt_temp.loc[
            history_ambt_temp[Cols.date] == row[Cols.date_time].strftime(TimeForms.date)
        ].iloc[0]["max_temp"]
        Day_1_min = history_ambt_temp.loc[
            history_ambt_temp[Cols.date]
            == (row[Cols.date_time] + pd.Timedelta(days=1)).strftime(TimeForms.date)
        ].iloc[0]["min_temp"]
        dt_2am_s = (
            row[Cols.date_time].time().hour * 3600
            + row[Cols.date_time].time().minute * 60
            + row[Cols.date_time].time().second
            - 7200
        )
        # delta time to 2am that day, in seconds
        ambt_temp = (
            Day_0_min
            + (Day_1_min - Day_0_min) * (dt_2am_s / Constants.seconds_in_day)
            + (Day_0_max - (Day_0_min + Day_1_min) / 2)
            / 2
            * (
                math.sin(
                    dt_2am_s / Constants.seconds_in_day * 2 * math.pi - math.pi / 2
                )
                + 1
            )
        )  # 根据三角函数计算出当前时刻的温度
        return ambt_temp

    df[Cols.ambt_temp] = df.apply(lambda row: fill_ambt(row, history_ambt_temp), axis=1)
    return df
