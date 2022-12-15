from loader import Loader
from plotter import *
import seaborn as sns
from thermal_model.configs import *


class Initial_delta_temp_histplot(Plotter):
    """主要用于展示最初读取数据并预处理完成后，分析展示原始的温度差分结果

    Args:
        Plotter (_type_): _description_
    """

    def __init__(
        self,
        df_thermal_model_data_raw,  # ThermalModelData().load()
        label="Thermal model",
        title="经过小波变换之后原始数据中的温度差分",
        title_plot=True,
    ) -> None:
        super().__init__(label, title, num_subplot=1, title_plot=title_plot)
        self.df_thermal_model_data_raw = df_thermal_model_data_raw

    def plot(self):
        sns.histplot(self.df_thermal_model_data_raw[Cols.delta_temp], bins=100)
        plt.xlabel("温度差分" + r"$(^\circ C)$")
        plt.ylabel("出现频次")


class Initial_delta_temp_pairplot(Plotter):
    """原始数据中的各相关数据的配对关系图

    Args:
        Plotter (_type_): _description_
    """

    def __init__(
        self,
        df_thermal_model_data_raw,  # ThermalModelData().load()
        label="Thermal model",
        title="原始数据中相关项的配对分析",
        num_subplot=1,
        title_plot=False,
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)
        self.df_thermal_model_data_raw = df_thermal_model_data_raw

    def plot(self):
        self.df_thermal_model_data_raw.rename(
            columns={
                Cols.cell_voltage: "电解电压",
                Cols.current_density: "电流密度",
                Cols.temp_out: "出口温度",
                Cols.lye_temp: "入口温度",
                Cols.lye_flow: "碱液流量",
                Cols.delta_temp: "温度差分",
            },
            inplace=True,
        )
        sns.pairplot(
            data=self.df_thermal_model_data_raw[
                [
                    "电解电压",
                    "电流密度",
                    "出口温度",
                    "入口温度",
                    "碱液流量",
                    "温度差分",
                ]
            ],
        )


class Model_input_data_pairplot(Plotter):
    def __init__(
        self,
        df_thermal_model_data_input,  # generate_model_input(df_thermal_model_data_raw)
        label="Thermal model",
        title="模型输入数据中相关项的配对分析",
        num_subplot=1,
        title_plot=False,
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)
        self.df_thermal_model_data_input = df_thermal_model_data_input

    def plot(self):
        sns.pairplot(
            self.df_thermal_model_data_input[
                thermal_model_input_cols + thermal_model_output_cols
            ]
        )
    
class Model_default_polarization_curve(Plotter):
    def __init__(
        self, 
        current_list,
        voltage_list,
        label="Thermal model", 
        title="电解槽极化曲线", 
        num_subplot=1, 
        title_plot=True
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)
        self.current_list = np.squeeze(current_list)
        self.voltage_list = np.squeeze(voltage_list)

    def plot(self):
        plt.plot(self.current_list,self.voltage_list)


class Thermal_model_regression_scatter(Plotter):
    def __init__(
        self, 
        model_target,
        model_predict,
        label="Thermal model", 
        title_model="随机森林", # 也可以是线性回归
        num_subplot=1, 
        title_plot=True
    ) -> None:
        title = "使用{}进行回归分析的误差结果".format(title_model)
        super().__init__(label, title, num_subplot, title_plot)
        self.model_target = np.array(model_target)
        self.model_predict = np.array(model_predict)
    
    def plot(self):
        plt.scatter(
            self.model_target,
            self.model_predict
        )
        minimum = min(self.model_target)
        maximum = max(self.model_predict)
        plt.plot(
            [minimum,maximum],
            [minimum,maximum],
            'r'
        )

class Thermal_model_regression_error_histplot(Plotter):
    def __init__(
        self, 
        model_target,
        error,
        label="Thermal model", 
        title_model="随机森林", # 也可以是线性回归
        num_subplot=1, 
        title_plot=True
    ) -> None:
        title = "使用{}进行回归分析的误差统计结果".format(title_model)
        super().__init__(label, title, num_subplot, title_plot)
        self.model_target = model_target
        self.error = error
    
    def plot(self):
        _ = plt.hist(
            self.model_target,bins = 1000
        )
        _ = plt.hist(
            self.error,bins = 1000
        )
        plt.legend(['regression target','regression error'])
        plt.plot([0,0],[0,2000],'r')
        plt.ylim([0,2000])
        plt.xlim([-0.5,0.5])
        
class Thermal_model_regression_cumulative_error_plot(Plotter):
    def __init__(
        self, 
        model_target,
        model_predict,
        label="Thermal model", 
        title_model="随机森林", # 也可以是线性回归
        num_subplot=1, 
        title_plot=True
    ) -> None:
        title = "使用{}进行回归的结果累计误差显示".format(title_model)
        super().__init__(label, title, num_subplot, title_plot)
        self.model_target = model_target
        self.model_predict = model_predict
    
    def plot(self):
        plt.plot(
            np.cumsum(
                self.model_target
            )
        )
        plt.plot(
            np.cumsum(
                self.model_predict
            )
        )
        plt.legend(['regression target','regression prediction'])

class Model_polarization_different_lye_temperature(Plotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="不同碱液入口温度下电解槽极化曲线", 
        num_subplot=1, 
        title_plot=True
    ) -> None:
        from thermal_model.electrolyzer import Electrolyzer
        super().__init__(label, title, num_subplot, title_plot)
        self.electrolyzer = Electrolyzer()
    
    def plot(self):
        lye_temperature_list = range(35,95,10)
        for lye_temperature in lye_temperature_list:
            (
                current_list,
                voltage_list,
                power_list,
                temperature_list
            ) = self.electrolyzer.get_polarization(
                lye_flow=1.5,
                lye_temperature=lye_temperature,
                current_max=2000,
                ambient_temperature=15,
            )
            plt.plot(
                np.array(current_list) / self.electrolyzer.active_surface_area,
                voltage_list,
                label = r'${} ^\circ C$'.format(
                    lye_temperature
                )
            )
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        plt.ylabel('Electrolyzer stack voltage (V)')
        plt.legend()
        plt.show()
