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
                    
        temperature_maximum = max(temperature_matrix.flatten())
        temperature_minimum = min(temperature_matrix.flatten())
        contour_levels = np.arange(
            temperature_minimum,
            temperature_maximum,
            (temperature_maximum - temperature_minimum)/10
        )
        contour_figure = plt.contourf(
            np.array(current_range) / self.electrolyzer.active_surface_area,
            lye_temperature_range,
            temperature_matrix,
            contour_levels,
            origin = 'upper'
        )
        plt.clabel(contour_figure, colors="w", fmt="%2.0f", fontsize=12)
        plt.xlabel(r'$Current\ density (A/m^2)$')
        plt.ylabel(r'$Lye inlet temperature (^\circ C)$')
        plt.colorbar(
            contour_figure
        )
        # 画出额定功率点
        plt.text(
            OperatingCondition.Rated.current_density+PlotterOffset.Marker.Cross.Subplot4.current_density,
            OperatingCondition.Rated.lye_temperature+PlotterOffset.Marker.Cross.Subplot4.lye_temperature,
            '+',
            color=OperatingCondition.Rated.color,
            fontdict={
                'size':PlotterOffset.Marker.Cross.Subplot4.font_size
            }
        )    
        plt.text(
            OperatingCondition.Rated.current_density-PlotterOffset.Marker.Cross.Subplot4.current_density,
            OperatingCondition.Rated.lye_temperature-PlotterOffset.Marker.Cross.Subplot4.lye_temperature,
            str(np.round(temperature_default,1) )+ r'$^\circ C$',
            color=OperatingCondition.Rated.color,
            fontdict={
                'size':PlotterOffset.Marker.Cross.Subplot4.font_size
            }
        )    
        # 画出最优工况点
        plt.text(
            OperatingCondition.Optimal.current_density+PlotterOffset.Marker.Cross.Subplot4.current_density,
            OperatingCondition.Optimal.lye_temperature+PlotterOffset.Marker.Cross.Subplot4.lye_temperature,
            '+',
            color=OperatingCondition.Rated.color,
            fontdict={
                'size':PlotterOffset.Marker.Cross.Subplot4.font_size
            }
        )    
        plt.text(
            OperatingCondition.Optimal.current_density-PlotterOffset.Marker.Cross.Subplot4.current_density,
            OperatingCondition.Optimal.lye_temperature-PlotterOffset.Marker.Cross.Subplot4.lye_temperature,
            str(np.round(temperature_optimal,1) )+ r'$^\circ C$',
            color=OperatingCondition.Rated.color,
            fontdict={
                'size':PlotterOffset.Marker.Cross.Subplot4.font_size
            }
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
            for ambient_temperature in ambient_temperature_range:
                temperature_cur = self.electrolyzer.temperature_thermal_balance_current(
                    ambient_temperature=ambient_temperature,
                    lye_flow= lye_flow,
                    lye_temperature = lye_temperature,
                    current=current
                )
                temperature_list.append(temperature_cur)
            plt.plot(
                ambient_temperature_range,
                temperature_list,
                label = np.round(lye_flow,1)
            )
        plt.xlabel(r'$Ambient\ temperature (^\circ C)$')
        plt.ylabel(r'$Outlet temperature (^\circ C)$')
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
                    
        temperature_maximum = max(temperature_matrix.flatten())
        temperature_minimum = min(temperature_matrix.flatten())
        contour_levels = np.arange(
            temperature_minimum,
            temperature_maximum,
            (temperature_maximum - temperature_minimum)/10
        )
        contour_figure = plt.contourf(
            np.array(current_range) / self.electrolyzer.active_surface_area,
            lye_temperature_range,
            temperature_matrix,
            contour_levels,
            origin = 'upper'
        )
        plt.clabel(contour_figure, colors="w", fmt="%2.0f", fontsize=12)
        plt.xlabel(r'$Current\ density (A/m^2)$')
        plt.ylabel(r'$Lye inlet temperature (^\circ C)$')
        plt.colorbar(
            contour_figure
        )
        # 画出额定功率点
        plt.text(
            OperatingCondition.Rated.current_density+PlotterOffset.Marker.Cross.Subplot4.current_density,
            OperatingCondition.Rated.lye_temperature+PlotterOffset.Marker.Cross.Subplot4.lye_temperature,
            '+',
            color=OperatingCondition.Rated.color,
            fontdict={
                'size':PlotterOffset.Marker.Cross.Subplot4.font_size
            }
        )    
        plt.text(
            OperatingCondition.Rated.current_density-PlotterOffset.Marker.Cross.Subplot4.current_density,
            OperatingCondition.Rated.lye_temperature-PlotterOffset.Marker.Cross.Subplot4.lye_temperature,
            str(np.round(temperature_default,1) )+ r'$^\circ C$',
            color=OperatingCondition.Rated.color,
            fontdict={
                'size':PlotterOffset.Marker.Cross.Subplot4.font_size
            }
        )    
        # 画出最优工况点
        plt.text(
            OperatingCondition.Optimal.current_density+PlotterOffset.Marker.Cross.Subplot4.current_density,
            OperatingCondition.Optimal.lye_temperature+PlotterOffset.Marker.Cross.Subplot4.lye_temperature,
            '+',
            color=OperatingCondition.Rated.color,
            fontdict={
                'size':PlotterOffset.Marker.Cross.Subplot4.font_size
            }
        )    
        plt.text(
            OperatingCondition.Optimal.current_density-PlotterOffset.Marker.Cross.Subplot4.current_density,
            OperatingCondition.Optimal.lye_temperature-PlotterOffset.Marker.Cross.Subplot4.lye_temperature,
            str(np.round(temperature_optimal,1) )+ r'$^\circ C$',
            color=OperatingCondition.Rated.color,
            fontdict={
                'size':PlotterOffset.Marker.Cross.Subplot4.font_size
            }
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
        plt.ylabel(r'$Temperature difference (^\circ C)$')
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
            y1_title= r'$Temperature difference (^\circ C)$',
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
            y1_title= r'$Temperature difference (^\circ C)$',
            y2_title='Stack voltage (V)',
        )
        ax2.set_ylim([60,70])
        ax2.set_yticks(range(60,71))
        ax1.set_ylim([10,60])