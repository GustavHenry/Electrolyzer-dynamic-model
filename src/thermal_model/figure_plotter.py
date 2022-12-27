from loader import Loader
from plotter import *
import seaborn as sns
from thermal_model.configs import *
from thermal_model.electrolyzer import Electrolyzer

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
                lye_flow=self.electrolyzer.default_lye_flow,
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
        plt.legend(
            title = 'Lye inlet temperature'
        )

class Model_faraday_efficiency_different_lye_temperature(Plotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="不同碱液入口温度下得电解槽法拉第效率响应曲线", 
        num_subplot=1, 
        title_plot=True
    ) -> None:
        
        super().__init__(label, title, num_subplot, title_plot)
        self.electrolyzer = Electrolyzer()
    
    def plot(self):
        temperature_list = range(35,107,20) # 只考虑不同出口温度下的法拉第效率
        current_max = 2000

        for temperature in temperature_list:
            current_list = range(0,current_max,current_max//100)
            faraday_efficiency_list = []
            for current in current_list:
                faraday_efficiency_cur = self.electrolyzer.faraday_efficiency_current(
                    current=current,
                    temperature=temperature
                )
                faraday_efficiency_list.append(faraday_efficiency_cur)
            plt.plot(
                np.array(current_list)/self.electrolyzer.active_surface_area,
                faraday_efficiency_list,
                label = r'${} ^\circ C$'.format(
                    temperature
                )
            )
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        plt.ylabel('Electrolyzer stack voltage')
        plt.legend(
            title = 'Outlet temperature'
        )

class Model_output_temperature_different_lye_temperature(QuadroPlotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="不同碱液入口温度下电解槽出口温度", 
        num_subplot=4, 
        title_plot=True
    ) -> None:
        
        super().__init__(label, title, num_subplot, title_plot)
        self.electrolyzer = Electrolyzer()
    
    def plot_1(self):
        #不同碱液入口温度下电解槽出口温度

        lye_temperature_range = range(
            OperatingRange.Contour.Lye_temperature.left,
            OperatingRange.Contour.Lye_temperature.right,
            OperatingRange.Contour.Lye_temperature.step
        )
        current_range = range(
            OperatingRange.Contour.Current.left,
            OperatingRange.Contour.Current.right,
            OperatingRange.Contour.Current.step
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        lye_flow = OperatingCondition.Default.lye_flow
        temperature_matrix = np.ones(
            (
                len(lye_temperature_range),
                len(current_range)
            )
        ) # 出口温度
        for i in range(len(current_range)):
            for j in range(len(lye_temperature_range)):
                1
                temperature_matrix[j,i] = (
                    self.electrolyzer.temperature_thermal_balance_current(
                        ambient_temperature=ambient_temperature,
                        lye_flow=lye_flow,
                        lye_temperature=lye_temperature_range[j],
                        current = current_range[i]
                    )
                )
                if (
                    lye_temperature_range[j] == OperatingCondition.Rated.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Rated.current
                ):
                    temperature_default = temperature_matrix[j,i]
                if (
                    lye_temperature_range[j] == OperatingCondition.Optimal.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Optimal.current
                ):
                    temperature_optimal = temperature_matrix[j,i]
        self.plot_contour_map_with_2_points(
            matrix= temperature_matrix,
            x_range=np.array(current_range) ,
            y_range=lye_temperature_range,
            value_default=temperature_default,
            value_optimal=temperature_optimal,
        )

    def plot_2(self):
        current_range = range(
            OperatingRange.Contour.Current.left,
            OperatingRange.Contour.Current.right,
            OperatingRange.Contour.Current.step*10
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        temperature = 90
        lye_flow_range = np.arange(
            OperatingRange.Contour.Lye_flow.left,
            OperatingRange.Contour.Lye_flow.right,
            OperatingRange.Contour.Lye_flow.step
            )

        for lye_flow in lye_flow_range:
            lye_temperature_list = []
            for current in current_range:
                lye_temperature_cur = self.electrolyzer.lye_temperature_for_given_condition_current(
                    current=current,
                    lye_flow=lye_flow,
                    temperature=temperature,
                    ambient_temperature=ambient_temperature,
                )
                lye_temperature_list.append(lye_temperature_cur)
            plt.plot(
                self.electrolyzer.current_2_density(current_range),
                lye_temperature_list,
                label = np.round(
                    lye_flow,
                    2
                )
            )
        plt.xlabel(r'$Current\ density (A/m^2)$')
        plt.ylabel(r'$Lye\ inlet\ temperature (^\circ C)$')
        plt.ylim([
            OperatingRange.Contour.Lye_temperature.left,
            OperatingRange.Contour.Lye_temperature.right
        ])
        plt.legend(
            title = r'$Lye\ flow (m^3/h)$',
            loc = 'best'
        )
        

    def plot_3(self):
        # 最优工况点随碱液流量的变化

        lye_flow_range = np.arange(
            OperatingRange.Contour.Lye_flow.left,
            OperatingRange.Contour.Lye_flow.right,
            OperatingRange.Contour.Lye_flow.step
            )
        current = OperatingCondition.Optimal.current
        lye_temperature = OperatingCondition.Optimal.lye_temperature
        temperature_list = []
        voltage_list = []
        for lye_flow in lye_flow_range:
            temperature_cur = self.electrolyzer.temperature_thermal_balance_current(
                ambient_temperature= OperatingCondition.Default.ambient_temperature,
                lye_flow= lye_flow,
                lye_temperature = lye_temperature,
                current=current
            )
            voltage_cur = self.electrolyzer.polar_current_lh(
                current = current,
                temperature=temperature_cur
            )
            temperature_list.append(temperature_cur)
            voltage_list.append(voltage_cur)
        
        ax1,ax2 = self.plot_double_y_axis(
            x = lye_flow_range,
            y1 = temperature_list,
            y2 = voltage_list,
            x_title = r'$Lye\ flow (m^3/h)$',
            y1_title= r'$Outlet\ temperature (^\circ C)$',
            y2_title='Stack voltage (V)',
        )
        ax2.set_ylim([60,65])
        # ax2.set_yticks(range(66,71))
        ax1.set_ylim([55,70])

    
    def plot_4(self):
        # 额定点工况温度随碱液流量的变化

        lye_flow_range = np.arange(
            OperatingRange.Contour.Lye_flow.left,
            OperatingRange.Contour.Lye_flow.right,
            OperatingRange.Contour.Lye_flow.step
            )
        current = OperatingCondition.Rated.current
        lye_temperature = OperatingCondition.Rated.lye_temperature
        temperature_list = []
        voltage_list = []
        for lye_flow in lye_flow_range:
            temperature_cur = self.electrolyzer.temperature_thermal_balance_current(
                ambient_temperature= self.electrolyzer.default_ambient_temperature,
                lye_flow= lye_flow,
                lye_temperature = lye_temperature,
                current=current
            )
            voltage_cur = self.electrolyzer.polar_current_lh(
                current = current,
                temperature=temperature_cur
            )
            temperature_list.append(temperature_cur)
            voltage_list.append(voltage_cur)
        
        ax1,ax2 = self.plot_double_y_axis(
            x = lye_flow_range,
            y1 = temperature_list,
            y2 = voltage_list,
            x_title = r'$Lye\ flow (m^3/h)$',
            y1_title= r'$Outlet\ temperature (^\circ C)$',
            y2_title='Stack voltage (V)',
        )
        ax2.set_ylim([60,70])
        ax2.set_yticks(range(60,71))
        ax1.set_ylim([70,120])
        

class Model_output_input_temperature_delta(QuadroPlotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="不同碱液入口温度下电解槽出入口温度差", 
        num_subplot=4, 
        title_plot=False
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)
        self.electrolyzer = Electrolyzer()
    
    def plot_1(self):
        #不同碱液入口温度下电解槽出口温度

        lye_temperature_range = range(
            OperatingRange.Contour.Lye_temperature.left,
            OperatingRange.Contour.Lye_temperature.right,
            OperatingRange.Contour.Lye_temperature.step
        )
        current_range = range(
            OperatingRange.Contour.Current.left,
            OperatingRange.Contour.Current.right,
            OperatingRange.Contour.Current.step
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        lye_flow = OperatingCondition.Default.lye_flow
        temperature_matrix = np.ones(
            (
                len(lye_temperature_range),
                len(current_range)
            )
        ) # 出口温度
        for i in range(len(current_range)):
            for j in range(len(lye_temperature_range)):
                1
                temperature_matrix[j,i] = (
                    self.electrolyzer.temperature_thermal_balance_current(
                        ambient_temperature=ambient_temperature,
                        lye_flow=lye_flow,
                        lye_temperature=lye_temperature_range[j],
                        current = current_range[i]
                    ) - lye_temperature_range[j]
                )
                if (
                    lye_temperature_range[j] == OperatingCondition.Rated.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Rated.current
                ):
                    temperature_default = temperature_matrix[j,i]
                if (
                    lye_temperature_range[j] == OperatingCondition.Optimal.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Optimal.current
                ):
                    temperature_optimal = temperature_matrix[j,i]
        self.plot_contour_map_with_2_points(
            matrix= temperature_matrix,
            x_range=np.array(current_range)  / self.electrolyzer.active_surface_area,
            y_range=lye_temperature_range,
            value_default=temperature_default,
            value_optimal=temperature_optimal,
        )

    def plot_2(self):
        # 额定点工况温度随的变化

        # lye_flow = OperatingCondition.Default.lye_flow
        ambient_temperature_range = range(
            OperatingRange.Contour.Ambient_temperature.left,
            OperatingRange.Contour.Ambient_temperature.right,
            OperatingRange.Contour.Ambient_temperature.step
            )
        current = OperatingCondition.Rated.current
        lye_temperature = OperatingCondition.Rated.lye_temperature
        
        lye_flow_range = np.arange(
            OperatingRange.Contour.Lye_flow.left,
            OperatingRange.Contour.Lye_flow.right,
            OperatingRange.Contour.Lye_flow.step
            )
        for lye_flow in lye_flow_range:
            temperature_list = []
            temperature_delta_list = []
            for ambient_temperature in ambient_temperature_range:
                temperature_cur = self.electrolyzer.temperature_thermal_balance_current(
                    ambient_temperature=ambient_temperature,
                    lye_flow= lye_flow,
                    lye_temperature = lye_temperature,
                    current=current
                )
                temperature_list.append(temperature_cur)
                temperature_delta_list.append(
                    temperature_cur - lye_temperature
                )
            plt.plot(
                ambient_temperature_range,
                temperature_delta_list,
                label = np.round(lye_flow,1)
            )
        plt.xlabel(r'$Ambient\ temperature (^\circ C)$')
        plt.ylabel(r'$Temperature\ difference (^\circ C)$')
        # plt.ylim([85,90])
        plt.legend(
            title = r'$Lye\ flow (m^3/h)$',
            loc = 'upper right'
        )

    def plot_3(self):
        # 最优工况点随碱液流量的变化

        lye_flow_range = np.arange(
            OperatingRange.Contour.Lye_flow.left,
            OperatingRange.Contour.Lye_flow.right,
            OperatingRange.Contour.Lye_flow.step
            )
        current = OperatingCondition.Optimal.current
        lye_temperature = OperatingCondition.Optimal.lye_temperature
        temperature_list = []
        voltage_list = []
        temperature_delta_list = []
        for lye_flow in lye_flow_range:
            temperature_cur = self.electrolyzer.temperature_thermal_balance_current(
                ambient_temperature= OperatingCondition.Default.ambient_temperature,
                lye_flow= lye_flow,
                lye_temperature = lye_temperature,
                current=current
            )
            voltage_cur = self.electrolyzer.polar_current_lh(
                current = current,
                temperature=temperature_cur
            )
            temperature_list.append(temperature_cur)
            temperature_delta_list.append(
                temperature_cur - lye_temperature
            )
            voltage_list.append(voltage_cur)
        
        ax1,ax2 = self.plot_double_y_axis(
            x = lye_flow_range,
            y1 = temperature_delta_list,
            y2 = voltage_list,
            x_title = r'$Lye\ flow (m^3/h)$',
            y1_title= r'$Temperature\ difference (^\circ C)$',
            y2_title='Stack voltage (V)',
        )
        ax2.set_ylim([60,65])
        # ax2.set_yticks(range(66,71))
        ax1.set_ylim([-5,10])

    
    def plot_4(self):
        # 额定点工况温度随碱液流量的变化

        lye_flow_range = np.arange(
            OperatingRange.Contour.Lye_flow.left,
            OperatingRange.Contour.Lye_flow.right,
            OperatingRange.Contour.Lye_flow.step
            )
        current = OperatingCondition.Rated.current
        lye_temperature = OperatingCondition.Rated.lye_temperature
        temperature_list = []
        voltage_list = []
        temperature_delta_list = []
        for lye_flow in lye_flow_range:
            temperature_cur = self.electrolyzer.temperature_thermal_balance_current(
                ambient_temperature= self.electrolyzer.default_ambient_temperature,
                lye_flow= lye_flow,
                lye_temperature = lye_temperature,
                current=current
            ) 
            voltage_cur = self.electrolyzer.polar_current_lh(
                current = current,
                temperature=temperature_cur
            )
            temperature_list.append(temperature_cur)
            voltage_list.append(voltage_cur)
            temperature_delta_list.append(
                temperature_cur - lye_temperature
            )
        ax1,ax2 = self.plot_double_y_axis(
            x = lye_flow_range,
            y1 = temperature_delta_list,
            y2 = voltage_list,
            x_title = r'$Lye\ flow (m^3/h)$',
            y1_title= r'$Temperature\ difference (^\circ C)$',
            y2_title='Stack voltage (V)',
        )
        ax2.set_ylim([60,70])
        ax2.set_yticks(range(60,71))
        ax1.set_ylim([10,60])

class Model_cooling_power_requirement(QuadroPlotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="不同碱液流量下的电解槽冷却功率需求", 
        num_subplot=4, 
        title_plot=False
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)
        self.electrolyzer = Electrolyzer()

    def plot_1(self):
        lye_temperature_range = range(
            OperatingRange.Contour.Lye_temperature.left,
            OperatingRange.Contour.Lye_temperature.right,
            OperatingRange.Contour.Lye_temperature.step
        )
        current_range = range(
            OperatingRange.Contour.Current.left,
            OperatingRange.Contour.Current.right,
            OperatingRange.Contour.Current.step
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        lye_flow = OperatingCondition.Default.lye_flow
        cooling_power_matrix = np.ones(
            (
                len(lye_temperature_range),
                len(current_range)
            )
        ) # 冷却功率需求
        for i in range(len(current_range)):
            for j in range(len(lye_temperature_range)):
                temperature_thermal_balance_cur  = (
                    self.electrolyzer.temperature_thermal_balance_current(
                        ambient_temperature=ambient_temperature,
                        lye_flow=lye_flow,
                        lye_temperature=lye_temperature_range[j],
                        current = current_range[i]
                    )
                )
                cooling_power_matrix[j,i] = self.electrolyzer.cooling_power_requirement(
                    temperature=temperature_thermal_balance_cur,
                    lye_temperature=lye_temperature_range[j],
                    lye_flow=lye_flow,
                )

                if (
                    lye_temperature_range[j] == OperatingCondition.Rated.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Rated.current
                ):
                    cooling_power_default = cooling_power_matrix[j,i]
                if (
                    lye_temperature_range[j] == OperatingCondition.Optimal.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Optimal.current
                ):
                    cooling_power_optimal = cooling_power_matrix[j,i]
                    
        self.plot_contour_map_with_2_points(
            matrix= cooling_power_matrix,
            x_range=np.array(current_range) / self.electrolyzer.active_surface_area ,
            y_range=lye_temperature_range,
            value_default=cooling_power_default,
            value_optimal=cooling_power_optimal,
            unit='kW'
        )
    
    def plot_2(self):
        # 这里画两条线，一条是冷却功率得绝对值，在不同碱液流量下的情况
        # 还有一条线是冷却功率需求占总功率得比例
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        current = OperatingCondition.Rated.current
        lye_temperature = OperatingCondition.Rated.lye_temperature
        current_range = range(
            OperatingRange.Cooling.Current.left,
            OperatingRange.Cooling.Current.right,
            OperatingRange.Cooling.Current.step,
        )
        
        lye_flow_range = np.arange(
            OperatingRange.Cooling.Lye_flow.left,
            OperatingRange.Cooling.Lye_flow.right,
            OperatingRange.Cooling.Lye_flow.step
            )
        self.cooling_power_ratio_list = []
        self.lye_flow_list = []
        for lye_flow in lye_flow_range:
            cooling_power_list = []
            cooling_power_ratio_list = []
            for current in current_range:
                temperature_cur = self.electrolyzer.temperature_thermal_balance_current(
                    ambient_temperature=ambient_temperature,
                    lye_flow= lye_flow,
                    lye_temperature = lye_temperature,
                    current=current
                )
                voltage_cur = self.electrolyzer.polar_current_lh(
                    current = current,
                    temperature=temperature_cur
                )
                cooling_power_cur = self.electrolyzer.cooling_power_requirement(
                    temperature=temperature_cur,
                    lye_temperature=lye_temperature,
                    lye_flow=lye_flow
                )
                cooling_power_ratio_cur = cooling_power_cur / current / voltage_cur * 1000 *100
                cooling_power_list.append(cooling_power_cur)
                cooling_power_ratio_list.append(cooling_power_ratio_cur)
            plt.plot(
                np.array(current_range)/self.electrolyzer.active_surface_area,
                cooling_power_list,
                label = np.round(lye_flow,2)
            )
            self.cooling_power_ratio_list.append(cooling_power_ratio_list)
            self.lye_flow_list.append(lye_flow)
        self.current_range = current_range
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        plt.ylabel('Cooling power requirement (kW)')
        plt.ylim([-15,40])
        plt.legend(
            title = r'$Lye\ flow (m^3/h)$',
            loc = 'upper left'
        )
        

    def plot_3(self):
        lye_flow_range = np.arange(
            OperatingRange.Cooling.Lye_flow.left,
            OperatingRange.Cooling.Lye_flow.right,
            OperatingRange.Cooling.Lye_flow.step/3
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature

        cooling_power_list_optimal = []
        cooling_power_list_rated = []
        for lye_flow in lye_flow_range:
            current = OperatingCondition.Optimal.current
            lye_temperature = OperatingCondition.Optimal.lye_temperature
            cooling_power_cur_optimal = self.electrolyzer.cooling_power_requirement(
                temperature=self.electrolyzer.temperature_thermal_balance_current(
                    ambient_temperature=ambient_temperature,
                    lye_flow= lye_flow,
                    lye_temperature = lye_temperature,
                    current=current
                ),
                lye_temperature=lye_temperature,
                lye_flow=lye_flow
            )
            current = OperatingCondition.Rated.current
            lye_temperature = OperatingCondition.Rated.lye_temperature
            cooling_power_cur_rated = self.electrolyzer.cooling_power_requirement(
                temperature=self.electrolyzer.temperature_thermal_balance_current(
                    ambient_temperature=ambient_temperature,
                    lye_flow= lye_flow,
                    lye_temperature = lye_temperature,
                    current=current
                ),
                lye_temperature=lye_temperature,
                lye_flow=lye_flow
            )
            cooling_power_list_optimal.append(cooling_power_cur_optimal)
            cooling_power_list_rated.append(cooling_power_cur_rated)
        ax1,ax2 = self.plot_double_y_axis(
            x = lye_flow_range,
            y1 = cooling_power_list_optimal,
            y2 = cooling_power_list_rated,
            x_title= r'$Lye\ flow (m^3/h)$',
            y1_title='Cooling power requirement for optimal condition (kW)',
            y2_title='Cooling power requirement for rated condition (kW)',
        )
        ax1.set_ylim([-10,10])
        ax1.set_yticks(range(-10,10,2))
        ax2.set_ylim([20,30])
    
    def plot_4(self):
        for idx in range(len(self.lye_flow_list)):
            plt.plot(
                np.array(self.current_range)/self.electrolyzer.active_surface_area,
                self.cooling_power_ratio_list[idx],
                label = np.round(self.lye_flow_list[idx],2)
            )
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        plt.ylabel('Cooling power requirement vs power input (%)')
        plt.ylim([-25,50])
        plt.legend(
            title = r'$Lye\ flow (m^3/h)$',
            loc = 'upper left'
        )