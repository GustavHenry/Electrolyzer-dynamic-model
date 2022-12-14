CACHE_DIR = "../.cache"
MODELS_DIR = "../models"
REPORTS_DIR = "../reports"
FIGURES_DIR = "../figures"
NOTEBOOKS_DIR = "../notebooks"


ENCODING = "utf-8-sig"  # csv的编码
COMPRESSION = "gzip"  # pickle的压缩格式


class DataDir:
    """一些常用的数据路径"""

    Raw = "../data/raw"
    Int = "../data/interim"
    Prc = "../data/processed"
    Raw_infrared = "../data/raw/InfraredImages"
    Raw_electrolyzer = "../data/raw/Electrolyzer"
    Raw_thermal_model = "../data/raw/ThermalModel"
    Int_thermal_model = "../data/interim/ThermalModel"
    Model_thermal_model = '../models/thermal_model'


class Files:
    """一些常用的，不会改变的数据文件"""

    history_ambient_temperature = "../data/raw/History_temperature_202001_202210.csv"
    aquarel_theme_scientific = "../configs/aquarel_theme_scientific.json"


class LoaderType:
    """loader 中不同的文件格式，主要会对读取文件产生区别"""

    dataframe = "dataframe"
    other = "other"


class Cache:
    """储存一些文件的中间缓存，用于统一缓存的读取与识别"""
    thermal_model_data_1101 = "thermal_model_data_1101" # 重写刚开始时，简单讲所有数据拼接在一起
    thermal_model_data_0102 = "thermal_modal_data_0102" # 抽选出适合作为热模型数据的准静态输入部分，并进行拼接


class Constants:
    seconds_in_hour = 3600
    seconds_in_day = 24 * 3600  # 每日中的秒数
    hours_in_year = 365*24 # 每日年中小时数
    num_cells = 34  # 电解槽片数
    active_area = 0.425  # 电解槽活性面积
    absolute_temp_delta = 273.15  # 绝对零度到零摄氏度的差值

    C_0 = 5.67  # W/(m2K4)黑体的辐射常数
    epsilon_steel = 0.95  # 电解槽表面的系统发射率
    rho_steel = 7900  # kg/m^3, the density of the structural material of the electrolyzer, ss304
    rho_alkaline = 1280  # kg/m^2, the density of the alkaline used in the electrolyzer
    specific_heat_capacity_steel = (
        0.5  # kJ/(kg*K), the specific heat capacity of the steel
    )
    specific_heat_capacity_alkaline = (
        2.2625  # kJ/(kg*K), the specific heat capacity of the alkaline
    )

    R = 96485 # 理想气体常数
    std_volume = 22.4 # mol/L
    weight_hydrogen = 2 # g/mol

class PolarizationLihao:
    r1 = 0.0001362
    r2 = -1.316e-06
    s1 = 0.06494
    s2 = 0.0013154
    s3 = -4.296e-06
    t1 = 0.1645
    t2 = -18.96
    t3 = 672.5

class Faraday:
    f11 = 1.067e4
    f12 = 101.1
    f21 = 0.989
    f22 = 7.641e-5


class Cols:
    """数据中会用到的列名"""

    time = "time"
    date = "date"
    date_time = "date_time"
    stack_voltage = "stack_voltage"  # 电解槽电压
    stack_current = "stack_current"  # 电解槽电流
    lye_flow = "lye_flow"  # 电解槽碱液流量
    lye_temp = "temp_in"  # 电解槽碱液入口温度
    sys_pressure = "sys_pressure"  # 电解槽工作压力
    o_temp = "temp_o"  # 电解槽出口氧侧温度
    h_temp = "temp_h"  # 电解槽出口氢侧温度

    o_in_h = "OTH"  # 氢中氧含量
    h_in_o = "HTO"  # 氧中氢含量

    temp_out = "temp_out"  # 电解槽出口氢氧温度平均
    delta_temp = "delta_temp"  # 当前时刻温度与上一时刻温度差
    current_density = "current_density"  # 电解槽工作的电流密度，电解槽面积为0.425平方米
    cell_voltage = "cell_voltage"  # 电解槽的小室电压，电解槽有34片
    ambt_temp = "ambt_temp"  # 环境温度需要使用时间和历史数据进行计算
    voltage_thermal_neutral = "voltage_thermal_neutral"  # 根据出口温度计算的热中性电压

    electric_heat = "electric_heat"  # 计算出的瞬时发热功率，单位应该是kW
    radiation_dissipation = "radiation_dissipation"  # 计算出的瞬时辐射散热功率，但并不是带量纲的结果
    input_lye_heat = "input_lye_heat"  # 进入流量乘以入口温度
    output_lye_heat = "output_lye_heat"  # 出口流量乘以出口温度


class TimeForms:
    # pandas.datetime .strftime()中的格式
    date = "%Y-%m-%d"  # 2021-09-24
    time = "%H:%M:%S"  # 09:20:20
    date_time = "%Y-%m-%d %H:%M:%S"  # 2021-09-24 09:20:20

class PlotterOffset:
    class Marker:
        class Cross:
            class Subplot4:
                current_density = -85
                lye_temperature = -1
                font_size = 16

