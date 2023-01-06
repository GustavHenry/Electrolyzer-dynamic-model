from loader import Loader
from plotter import *
import seaborn as sns
from thermal_model.configs import *
from thermal_model.electrolyzer import Electrolyzer
from thermal_model.data import ThermalModelData,generate_model_input
from thermal_model.thermal_model import fit_random_forest,model_estimator

class Initial_delta_temp_histplot(Plotter):
    """主要用于展示最初读取数据并预处理完成后，分析展示原始的温度差分结果

    Args:
        Plotter (_type_): _description_
    """

    def __init__(
        self,
        label="Thermal model",
        title="经过小波变换之后原始数据中的温度差分",
        title_plot=True,
    ) -> None:
        super().__init__(label, title, num_subplot=1, title_plot=title_plot)
        self.df_thermal_model_data_raw = ThermalModelData().load()

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
        label="Thermal model",
        title="原始数据中相关项的配对分析",
        num_subplot=1,
        title_plot=False,
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)
        self.df_thermal_model_data_raw = ThermalModelData().load()

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
        label="Thermal model",
        title="模型输入数据中相关项的配对分析",
        num_subplot=1,
        title_plot=False,
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)
        df_thermal_model_data_raw = ThermalModelData().load()
        self.df_thermal_model_data_input = generate_model_input(df_thermal_model_data_raw)

    def plot(self):
        sns.pairplot(
            self.df_thermal_model_data_input[
                thermal_model_input_cols + thermal_model_output_cols
            ]
        )
    
class Model_default_polarization_curve(Plotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="电解槽极化曲线", 
        num_subplot=1, 
        title_plot=True
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)
        (
            current_list,
            voltage_list,
            power_list,
            temperature_list
        ) = Electrolyzer().get_default_polarization()
        self.current_list = np.squeeze(current_list)
        self.voltage_list = np.squeeze(voltage_list)

    def plot(self):
        plt.plot(self.current_list,self.voltage_list)


class Thermal_model_regression_scatter(Plotter):
    def __init__(
        self, 
        label="Thermal model", 
        title_model="随机森林", # 也可以是线性回归
        num_subplot=1, 
        title_plot=True
    ) -> None:
        title = "使用{}进行回归分析的误差结果".format(title_model)
        super().__init__(label, title, num_subplot, title_plot)
        df_thermal_model_data_raw = ThermalModelData().load()
        df_thermal_model_data_input = generate_model_input(df_thermal_model_data_raw)

        model_random_forest,model_input,model_target = fit_random_forest(df_thermal_model_data_input,6)
        ( model_predict, error) = model_estimator(
            model_random_forest,model_input,model_target
        )
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
        label="Thermal model", 
        title_model="随机森林", # 也可以是线性回归
        num_subplot=1, 
        title_plot=True
    ) -> None:
        title = "使用{}进行回归分析的误差统计结果".format(title_model)
        super().__init__(label, title, num_subplot, title_plot)
        df_thermal_model_data_raw = ThermalModelData().load()
        df_thermal_model_data_input = generate_model_input(df_thermal_model_data_raw)

        model_random_forest,model_input,model_target = fit_random_forest(df_thermal_model_data_input,6)
        ( model_predict, error) = model_estimator(
            model_random_forest,model_input,model_target
        )
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
        label="Thermal model", 
        title_model="随机森林", # 也可以是线性回归
        num_subplot=1, 
        title_plot=True
    ) -> None:
        title = "使用{}进行回归的结果累计误差显示".format(title_model)
        super().__init__(label, title, num_subplot, title_plot)
        df_thermal_model_data_raw = ThermalModelData().load()
        df_thermal_model_data_input = generate_model_input(df_thermal_model_data_raw)

        model_random_forest,model_input,model_target = fit_random_forest(df_thermal_model_data_input,6)
        ( model_predict, error) = model_estimator(
            model_random_forest,model_input,model_target
        )
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


class Data_model_regression(DualPlotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="原始数据与模型拟合结果对比", 
        num_subplot=2, 
        title_plot=False
    ) -> None:
        # 大致包含：
        # Thermal_model_regression_scatter().save()
        # Thermal_model_regression_error_histplot().save()
        # Model_polarization_different_lye_temperature().save()
        # Model_faraday_efficiency_different_lye_temperature().save()
        super().__init__(label, title, num_subplot, title_plot)
        self.electrolyzer = Electrolyzer()

    def plot_1(self):
        df_thermal_model_data_raw = ThermalModelData().load()
        df_thermal_model_data_input = generate_model_input(df_thermal_model_data_raw)

        model_random_forest,model_input,model_target,score = fit_random_forest(
            df_thermal_model_data_input,
            4
        )
        ( model_predict, error) = model_estimator(
            model_random_forest,model_input,model_target
        )
        self.model_target = np.array(model_target)
        self.model_predict = np.array(model_predict)
        self.error = np.array(error)
        plt.scatter(
            self.model_target[:1],
            self.model_predict[:1],
            label = 'Target vs. prediction',
            # marker = 'x',
            # alpha=0.05
            color = self.color_list[0]
        )
        plt.scatter(
            self.model_target,
            self.model_predict,
            # label = 'Target vs. prediction',
            # marker = 'x',
            alpha=0.05,
            color = self.color_list[0]
        )
        minimum = min(self.model_target)
        maximum = max(self.model_predict)
        plt.plot(
            [-1,1],
            [-1,1],
            'r'
        )
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.xlabel('Standardized target')
        plt.ylabel('Model prediction')
        plt.legend(['Target vs. prediction'])
    
    def plot_2(self):
        df_thermal_model_data_raw = ThermalModelData().load()
        df_thermal_model_data_input = generate_model_input(df_thermal_model_data_raw)

        model_random_forest,model_input,model_target,score = fit_random_forest(
            df_thermal_model_data_input,
            5
        )

        ( model_predict, error) = model_estimator(
            model_random_forest,model_input,model_target
        )
        self.model_target = model_target
        self.model_predict = model_predict
        start = 18000
        end = 24200
        offset = 18
        x_range = np.array(range(0,end-start))*20
        plt.plot(
            x_range,
            np.cumsum(
                self.model_target[start:end]
            ) +offset
        )
        plt.plot(
            x_range,
            np.cumsum(
                self.model_predict[start:end]
            ) + offset
        )
        plt.legend(['regression target','regression prediction'])
        plt.xlabel('Time (s)')
        plt.ylabel(r'$Operating\ temperature (^\circ C)$')
        plt.ylim([0,100])
        Loader.save_model(
            model = model_random_forest,
            file_name='Random_forest',
            score=score
        )
        # plt.xticks(np.array(range(0,end-start))*20)


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


class Model_polarization_faraday(DualPlotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="不同温度下电解槽极化与法拉第效率", 
        num_subplot=2, 
        title_plot=False
    ) -> None:
        # 大致包含：
        # Thermal_model_regression_scatter().save()
        # Thermal_model_regression_error_histplot().save()
        # Model_polarization_different_lye_temperature().save()
        # Model_faraday_efficiency_different_lye_temperature().save()
        super().__init__(label, title, num_subplot, title_plot)
        self.electrolyzer = Electrolyzer()

    def plot_1(self):
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
                np.array(voltage_list) / self.electrolyzer.num_cells,
                label = r'${} ^\circ C$'.format(
                    lye_temperature
                )
            )
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        plt.ylabel('Electrolysis voltage (V)')
        plt.legend(
            title = 'Lye inlet temperature'
        )
        # plt.ylim([1.2,2.4])

    def plot_2(self):
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
        plt.ylim([0,1])
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        plt.ylabel('Faraday efficiency')
        plt.legend(
            title = 'Outlet temperature'
        )

class Data_demonstrate_model_fit(QuadroPlotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="原始数据展示与模型拟合结果", 
        num_subplot=4, 
        title_plot=False
    ) -> None:
        # 大致包含：
        # Thermal_model_regression_scatter().save()
        # Thermal_model_regression_error_histplot().save()
        # Model_polarization_different_lye_temperature().save()
        # Model_faraday_efficiency_different_lye_temperature().save()
        super().__init__(label, title, num_subplot, title_plot)
        self.electrolyzer = Electrolyzer()

    def plot_1(self):
        df_thermal_model_data_raw = ThermalModelData().load()
        df_thermal_model_data_input = generate_model_input(df_thermal_model_data_raw)

        model_random_forest,model_input,model_target = fit_random_forest(
            df_thermal_model_data_input,
            4
        )
        ( model_predict, error) = model_estimator(
            model_random_forest,model_input,model_target
        )
        self.model_target = np.array(model_target)
        self.model_predict = np.array(model_predict)
        self.error = np.array(error)
        plt.scatter(
            self.model_target[:1],
            self.model_predict[:1],
            label = 'Target vs. prediction',
            # marker = 'x',
            # alpha=0.05
            color = self.color_list[0]
        )
        plt.scatter(
            self.model_target,
            self.model_predict,
            # label = 'Target vs. prediction',
            # marker = 'x',
            alpha=0.05,
            color = self.color_list[0]
        )
        minimum = min(self.model_target)
        maximum = max(self.model_predict)
        plt.plot(
            [-1,1],
            [-1,1],
            'r'
        )
        plt.xlim([-1,1])
        plt.ylim([-1,1])
        plt.xlabel('Standardized target')
        plt.ylabel('Model prediction')
        plt.legend(['Target vs. prediction'])
    
    def plot_2(self):
        _ = plt.hist(
            self.model_target,
            bins = 1000
        )
        _ = plt.hist(
            self.error,
            bins = 1000
        )
        plt.legend(['regression target','regression error'])
        plt.plot([0,0],[0,2000],'r')
        plt.ylim([0,1250])
        plt.xlim([-0.5,0.5])

        plt.legend(['Regression target','Prediction error'])
        plt.ylabel('Frequency')
        plt.xlabel(r'$Temperature\ change\ (^\circ C / s)$')

    def plot_3(self):
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
                np.array(voltage_list) / self.electrolyzer.num_cells,
                label = r'${} ^\circ C$'.format(
                    lye_temperature
                )
            )
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        plt.ylabel('Electrolysis voltage (V)')
        plt.legend(
            title = 'Lye inlet temperature'
        )
        # plt.ylim([1.2,2.4])

    def plot_4(self):
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
        plt.ylim([0,1])
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        plt.ylabel('Faraday efficiency')
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
            x_range=np.array(current_range) /self.electrolyzer.active_surface_area,
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
            voltage_list.append(voltage_cur/self.electrolyzer.num_cells)
        
        ax1,ax2 = self.plot_double_y_axis(
            x = lye_flow_range,
            y1 = temperature_list,
            y2 = voltage_list,
            x_title = r'$Lye\ flow (m^3/h)$',
            y1_title= r'$Outlet\ temperature (^\circ C)$',
            y2_title='Cell voltage (V)',
        )
        ax2.set_ylim([1.2,2.2])
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
            voltage_list.append(voltage_cur/self.electrolyzer.num_cells)
        
        ax1,ax2 = self.plot_double_y_axis(
            x = lye_flow_range,
            y1 = temperature_list,
            y2 = voltage_list,
            x_title = r'$Lye\ flow (m^3/h)$',
            y1_title= r'$Outlet\ temperature (^\circ C)$',
            y2_title='Cell voltage (V)',
        )
        ax2.set_ylim([1.2,2.2])
        # ax2.set_yticks(range(60,71))
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
        lye_flow_list = [0.8,1.2,1.6]
        lye_flow_range = np.arange(
            OperatingRange.Cooling.Lye_flow.left,
            OperatingRange.Cooling.Lye_flow.right,
            OperatingRange.Cooling.Lye_flow.step
            )
        color_idx = 0
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        for lye_flow in lye_flow_list:
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
            
            ax1.plot(
                np.array(current_range)/self.electrolyzer.active_surface_area,
                cooling_power_list,
                label = 'lye flow = ' + str(np.round(lye_flow,2)) + r'$\ (m^3/h)$',
                marker = '.',
                color = self.color_list[color_idx]
            )
            
            ax2.plot(
                np.array(current_range)/self.electrolyzer.active_surface_area,
                cooling_power_ratio_list,
                label = 'Lye flow = ' + str(np.round(lye_flow,2)) + r'$\ (m^3/h)$',
                marker = 'x',
                color = self.color_list[color_idx]
            )
            color_idx +=1
        self.current_range = current_range
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        ax1.set_ylabel('Temperature maintenance requirement(kW)')
        ax2.set_ylabel('Temperature maintenance ratio(%)')
        ax1.set_ylim([-25,40])
        ax2.set_ylim([-25,80])
        ax1.legend(
            title ='Temperature maintenance\n requirement(kW)',
            loc = 'upper left'
        )
        ax2.legend(
            title = 'Temperature maintenance ratio(%)',
            loc = 'lower right'
        )
        plt.grid(visible=False)
        

    def plot_3(self):
        lye_flow_range = np.arange(
            OperatingRange.Cooling.Lye_flow.left,
            OperatingRange.Cooling.Lye_flow.right,
            OperatingRange.Cooling.Lye_flow.step/3
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        lye_temperature = OperatingCondition.Optimal.lye_temperature
        lye_temperature_list = [40,50,60,70,80]
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        color_idx = 0
        for lye_temperature in lye_temperature_list:
            cooling_power_list_optimal = []
            cooling_power_list_rated = []

            for lye_flow in lye_flow_range:
                current = OperatingCondition.Optimal.current
                
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
            ax1.plot(
                lye_flow_range,
                cooling_power_list_optimal,
                label =  str(np.round(lye_temperature,2)) + r'$\ ^\circ C$',
                marker = '.',
                color = self.color_list[color_idx]
            )
            
            ax2.plot(
                lye_flow_range,
                cooling_power_list_rated,
                label =  str(np.round(lye_temperature,2)) + r'$\ ^\circ C$',
                marker = 'x',
                color = self.color_list[color_idx]
            )
            color_idx +=1
        
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        ax1.set_ylabel('Requirement for optimal condition (kW)',)
        ax2.set_ylabel('Requirement for rated condition (kW)')

        ax1.legend(
            title ='Optimal condition,\n lye temperature = ',
            loc = 'lower left'
        )
        ax2.legend(
            title = 'Rated condition,\n lye temperature = ',
            loc = 'upper right'
        )
        plt.grid(visible=False)
        ax1.set_ylim([-10,30])
        # ax1.set_yticks(range(-10,10,2))
        ax2.set_ylim([0,40])
    
    def plot_4(self):
        cooling_efficiency_list = [0.15,0.25,0.35,0.45,0.55]
        current_range = range(
            OperatingRange.Contour.Current.left,
            OperatingRange.Contour.Current.right,
            OperatingRange.Contour.Current.step
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        lye_flow = OperatingCondition.Default.lye_flow
        lye_temperature = OperatingCondition.Default.lye_temperature
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        color_idx = 0
        for cooling_efficiency in cooling_efficiency_list:
            cooling_power_list = []
            cooling_power_ratio_list = []
            for current in current_range:
                power_cur,cooling_power_cur = self.electrolyzer.power_detail(
                    current=current,
                    lye_temperature=lye_temperature,
                    ambient_temperature=ambient_temperature,
                    lye_flow=lye_flow,
                    cooling_efficiency=cooling_efficiency
                )
                
                cooling_power_list.append(cooling_power_cur)
                cooling_power_ratio_list.append(
                    cooling_power_cur / power_cur * 100
                )
            ax1.plot(
                np.array(current_range) / self.electrolyzer.active_surface_area,
                cooling_power_list,
                label = cooling_efficiency,
                marker = '.',
                color = self.color_list[color_idx]
            )
            ax2.plot(
                np.array(current_range) / self.electrolyzer.active_surface_area,
                cooling_power_ratio_list,
                label = cooling_efficiency,
                marker = 'x',
                color = self.color_list[color_idx]
            )
            ax2.plot(
                [0,OperatingRange.Contour.Current.right/self.electrolyzer.active_surface_area],
                [0,0],
                color = 'grey',
                linestyle = '-.'
            )
            color_idx += 1
        plt.xlabel(r'$Current\ density\ (A/m^2)$')
        ax1.set_ylabel('Temperature maintenance power (kW)',)
        ax2.set_ylabel('Temperature maintenance ratio (%)')
        ax1.set_ylim([0,60])
        ax2.set_ylim([-25,30])
        ax1.legend(
            title ='Power,\n Cooling efficiency=',
            loc = 'lower left'
        )
        ax2.legend(
            title = 'Ratio,\n Cooling efficiency=',
            loc = 'upper right'
        )
        plt.grid(visible=False)


class Model_efficiency_hydrogen_cost(QuadroPlotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="不同工况下的电解槽高热值热效率与制氢能耗", 
        num_subplot=4, 
        title_plot=False
    ) -> None:
        """里面的1、3分别留给高热值热效率，2、4留给

        Args:
            label (str, optional): _description_. Defaults to "Thermal model".
            title (str, optional): _description_. Defaults to "不同工况下的电解槽高热值热效率与制氢能耗".
            num_subplot (int, optional): _description_. Defaults to 4.
            title_plot (bool, optional): _description_. Defaults to False.
        """
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
        efficiency_matrix = np.ones(
            (
                len(lye_temperature_range),
                len(current_range)
            )
        ) # 电解槽高热值热效率
        for i in range(len(current_range)):
            for j in range(len(lye_temperature_range)):
                efficiency_matrix[j,i] = self.electrolyzer.efficiency_current(
                    current=current_range[i],
                    ambient_temperature=ambient_temperature,
                    lye_flow=lye_flow,
                    lye_temperature=lye_temperature_range[j]
                )
                if (
                    lye_temperature_range[j] == OperatingCondition.Rated.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Rated.current
                ):
                    efficiency_default = efficiency_matrix[j,i]
                if (
                    lye_temperature_range[j] == OperatingCondition.Optimal.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Optimal.current
                ):
                    efficiency_optimal = efficiency_matrix[j,i]
        self.plot_contour_map_with_2_points(
            matrix=efficiency_matrix,
            x_range=np.array(current_range) / self.electrolyzer.active_surface_area,
            y_range=lye_temperature_range,
            value_default=efficiency_default,
            value_optimal=efficiency_optimal,
            unit = '%',
            value_min = 65,
            value_max=90
        )


    def plot_2(self):
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
        cost_matrix = np.ones(
            (
                len(lye_temperature_range),
                len(current_range)
            )
        ) # 电解槽高热值热效率
        for i in range(len(current_range)):
            for j in range(len(lye_temperature_range)):
                cost_matrix[j,i] = self.electrolyzer.hydrogen_cost_current(
                    current=current_range[i],
                    lye_flow=lye_flow,
                    lye_temperature=lye_temperature_range[j],
                    ambient_temperature=ambient_temperature,

                )

                if (
                    lye_temperature_range[j] == OperatingCondition.Rated.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Rated.current
                ):
                    cost_default = cost_matrix[j,i]
                if (
                    lye_temperature_range[j] == OperatingCondition.Optimal.lye_temperature
                ) and (
                    current_range[i]==OperatingCondition.Optimal.current
                ):
                    cost_optimal = cost_matrix[j,i]
        self.plot_contour_map_with_2_points(
            matrix=cost_matrix,
            x_range=np.array(current_range) / self.electrolyzer.active_surface_area,
            y_range=lye_temperature_range,
            value_default=cost_default,
            value_optimal=cost_optimal,
            unit = r'$kWh/Nm^3$',
            value_max=5.8,
            value_min=4.2
        )

    def plot_3(self):
        # 额定工作点和最优工作点的效率情况
        lye_flow_range = np.arange(
            OperatingRange.Cooling.Lye_flow.left,
            OperatingRange.Cooling.Lye_flow.right,
            OperatingRange.Cooling.Lye_flow.step/3
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        efficiency_list_optimal = []
        efficiency_list_rated = []

        for lye_flow in lye_flow_range:
            current = OperatingCondition.Optimal.current
            lye_temperature = OperatingCondition.Optimal.lye_temperature
            efficiency_cur_optimal = self.electrolyzer.efficiency_current(
                current=current,
                ambient_temperature=ambient_temperature,
                lye_flow=lye_flow,
                lye_temperature=lye_temperature
            )
            current = OperatingCondition.Rated.current
            lye_temperature = OperatingCondition.Rated.lye_temperature
            efficiency_cur_rated = self.electrolyzer.efficiency_current(
                current=current,
                ambient_temperature=ambient_temperature,
                lye_flow=lye_flow,
                lye_temperature=lye_temperature
            )
            efficiency_list_optimal.append(efficiency_cur_optimal)
            efficiency_list_rated.append(efficiency_cur_rated)
        plt.plot(
            lye_flow_range,
            efficiency_list_optimal,
            label = 'Optimal condition'
        )
        plt.plot(
            lye_flow_range,
            efficiency_list_rated,
            label = 'Rated condition'
        )
        plt.xlabel(
            r'$Lye\ flow (m^3/h)$',
        )
        plt.ylabel(
            'Electrolyzer efficiency (%)'
        )
        plt.ylim([70,85])
        plt.legend(
            title = 'Operating condition'
        )

    def plot_4(self):
        # 额定工作点和最优工作点的效率情况
        lye_flow_range = np.arange(
            OperatingRange.Cooling.Lye_flow.left,
            OperatingRange.Cooling.Lye_flow.right,
            OperatingRange.Cooling.Lye_flow.step/3
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        cost_list_optimal = []
        cost_list_rated = []

        for lye_flow in lye_flow_range:
            current = OperatingCondition.Optimal.current
            lye_temperature = OperatingCondition.Optimal.lye_temperature
            cost_cur_optimal = self.electrolyzer.hydrogen_cost_current(
                current=current,
                ambient_temperature=ambient_temperature,
                lye_flow=lye_flow,
                lye_temperature=lye_temperature
            )
            current = OperatingCondition.Rated.current
            lye_temperature = OperatingCondition.Rated.lye_temperature
            cost_cur_rated = self.electrolyzer.hydrogen_cost_current(
                current=current,
                ambient_temperature=ambient_temperature,
                lye_flow=lye_flow,
                lye_temperature=lye_temperature
            )
            cost_list_optimal.append(cost_cur_optimal)
            cost_list_rated.append(cost_cur_rated)
        plt.plot(
            lye_flow_range,
            cost_list_optimal,
            label = 'Optimal condition'
        )
        plt.plot(
            lye_flow_range,
            cost_list_rated,
            label = 'Rated condition'
        )
        plt.xlabel(
            r'$Lye\ flow (m^3/h)$',
        )
        plt.ylabel(
            r'$Hydrogen\ production\ cost\ (kWh/Nm^3)$'
        )
        plt.ylim([4.2,5.8])
        plt.legend(
            title = 'Operating condition'
        )

class Model_life_cycle_hydrogen_cost(HexaPlotter):
    def __init__(
        self, 
        label="Thermal model", 
        title="不同工况下电解槽全生命周期制氢成本", 
        num_subplot=6, 
        title_plot=False
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)
        # 还可以讨论整个生命周期的盈利？或者收入？
        
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
        electricity_price = LifeCycle.electricity_price
        electrolyzer_price = 250000
        cooling_efficiency = LifeCycle.cooling_efficiency
        heating_efficiency = LifeCycle.heating_efficiency
        cost_matrix = np.ones(
            (
                len(lye_temperature_range),
                len(current_range)
            )
        ) # 电解槽全生命周期制氢成本
        for i in range(len(current_range)):
            for j in range(len(lye_temperature_range)):

                current = current_range[i]
                lye_temperature = lye_temperature_range[j]
                cost_matrix[j,i] = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current=current,
                    ambient_temperature=ambient_temperature,
                    lye_flow=lye_flow,
                    lye_temperature=lye_temperature,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                )
                
        cost_default = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current=OperatingCondition.Rated.current,
                    lye_temperature=OperatingCondition.Rated.lye_temperature,
                    ambient_temperature=ambient_temperature,
                    lye_flow=lye_flow,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                )
        cost_optimal = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current=OperatingCondition.Optimal.current,
                    lye_temperature=OperatingCondition.Optimal.lye_temperature,
                    ambient_temperature=ambient_temperature,
                    lye_flow=lye_flow,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                )
        self.plot_contour_map_with_2_points(
            matrix=cost_matrix,
            x_range=np.array(current_range) / self.electrolyzer.active_surface_area,
            y_range=lye_temperature_range,
            value_default=cost_default,
            value_optimal=cost_optimal,
            unit=' $USD/kg',
            value_max=5.6,
            value_min=4.4
        )

    def plot_2(self):
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
        electricity_price = LifeCycle.electricity_price
        electrolyzer_price = 500000
        cooling_efficiency = LifeCycle.cooling_efficiency
        heating_efficiency = LifeCycle.heating_efficiency
        cost_matrix = np.ones(
            (
                len(lye_temperature_range),
                len(current_range)
            )
        ) # 电解槽全生命周期制氢成本
        for i in range(len(current_range)):
            for j in range(len(lye_temperature_range)):

                current = current_range[i]
                lye_temperature = lye_temperature_range[j]
                cost_matrix[j,i] = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current=current,
                    ambient_temperature=ambient_temperature,
                    lye_flow=lye_flow,
                    lye_temperature=lye_temperature,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                )
                
        cost_default = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current=OperatingCondition.Rated.current,
                    lye_temperature=OperatingCondition.Rated.lye_temperature,
                    ambient_temperature=ambient_temperature,
                    lye_flow=lye_flow,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                )
        cost_optimal = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current=OperatingCondition.Optimal.current,
                    lye_temperature=OperatingCondition.Optimal.lye_temperature,
                    ambient_temperature=ambient_temperature,
                    lye_flow=lye_flow,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                )
        self.plot_contour_map_with_2_points(
            matrix=cost_matrix,
            x_range=np.array(current_range) / self.electrolyzer.active_surface_area,
            y_range=lye_temperature_range,
            value_default=cost_default,
            value_optimal=cost_optimal,
            unit=' $/kg',
            value_max=5.6,
            value_min=4.4
        )
    
    def plot_3(self):
        # 两个工况点在不同碱液流量下的成本，最好也能包含不同的价格
        lye_flow_range = np.arange(
            OperatingRange.Cooling.Lye_flow.left,
            OperatingRange.Cooling.Lye_flow.right,
            OperatingRange.Cooling.Lye_flow.step/3
        )
        ambient_temperature = OperatingCondition.Default.ambient_temperature

        ambient_temperature = OperatingCondition.Default.ambient_temperature
        electricity_price = LifeCycle.electricity_price
        electrolyzer_price_list = [150000,250000,350000]
        cooling_efficiency = LifeCycle.cooling_efficiency
        heating_efficiency = LifeCycle.heating_efficiency
        color_idx = 0
        for electrolyzer_price in electrolyzer_price_list:
            cost_list_optimal = []
            cost_list_rated = []
            for lye_flow in lye_flow_range:
                cost_cur_optimal = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current= OperatingCondition.Optimal.current,
                    lye_temperature=OperatingCondition.Optimal.lye_temperature,
                    lye_flow=lye_flow,
                    ambient_temperature=ambient_temperature,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                )
                cost_cur_rated = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current= OperatingCondition.Default.current,
                    lye_temperature=OperatingCondition.Default.lye_temperature,
                    lye_flow=lye_flow,
                    ambient_temperature=ambient_temperature,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                )
                cost_list_optimal.append(cost_cur_optimal)
                cost_list_rated.append(cost_cur_rated)
            plt.plot(
                lye_flow_range,
                cost_list_optimal,
                label = 'Optimal condition, price = ${}'.format(
                    np.round(
                        electrolyzer_price*LifeCycle.RMB_2_USD,
                        1
                    )
                ),
                marker = '.',
                color = self.color_list[color_idx]
            )
            plt.plot(
                lye_flow_range,
                cost_list_rated,
                label = 'Rated condition, price = ${}'.format(
                    np.round(
                        electrolyzer_price*LifeCycle.RMB_2_USD,
                        1
                    )
                ),
                marker = 'x',
                color = self.color_list[color_idx]
            )
            color_idx += 1
        plt.xlabel(
            r'$Lye\ flow (m^3/h)$',
        )
        plt.ylabel(
            r'$Hydrogen\ production\ cost\ (\$/kg)$'
        )
        # plt.ylim([4.2,5.8])
        plt.legend(
            title = 'Operating condition'
        )
        plt.ylim([4.4,5.6])

    def plot_4_archived(self):
        # 两个工况点在不同电解槽价格下的氢气价格
        # 暂时不再进行此计算
        electrolyzer_price_range = range(
            LifeCycle.ElectrolyzerPriceRange.left,
            LifeCycle.ElectrolyzerPriceRange.right,
            LifeCycle.ElectrolyzerPriceRange.step
        )
        lye_flow_list = [0.8,1.2,1.6]
        ambient_temperature = OperatingCondition.Default.ambient_temperature

        ambient_temperature = OperatingCondition.Default.ambient_temperature
        electricity_price = LifeCycle.electricity_price
        cooling_efficiency = LifeCycle.cooling_efficiency
        heating_efficiency = LifeCycle.heating_efficiency
        color_idx = 0
        for lye_flow in lye_flow_list:
            
            cost_list_optimal = []
            cost_list_rated = []
            for electrolyzer_price in electrolyzer_price_range:
                cost_cur_optimal = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current= OperatingCondition.Optimal.current,
                    lye_temperature=OperatingCondition.Optimal.lye_temperature,
                    lye_flow=lye_flow,
                    ambient_temperature=ambient_temperature,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                )
                cost_cur_rated = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current= OperatingCondition.Default.current,
                    lye_temperature=OperatingCondition.Default.lye_temperature,
                    lye_flow=lye_flow,
                    ambient_temperature=ambient_temperature,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                )
                cost_list_optimal.append(cost_cur_optimal)
                cost_list_rated.append(cost_cur_rated)
            plt.plot(
                np.array(electrolyzer_price_range)*LifeCycle.RMB_2_USD,
                cost_list_optimal,
                label = 'Optimal condition, lye flow = {}'.format(lye_flow),
                marker = '.',
                color = self.color_list[color_idx]
            )
            if not lye_flow == 1.6:
                plt.plot(
                    np.array(electrolyzer_price_range)*LifeCycle.RMB_2_USD,
                    cost_list_rated,
                    label = 'Rated condition, lye flow = {}'.format(lye_flow),
                    marker = 'x',
                color = self.color_list[color_idx]
                )
            color_idx += 1
        plt.xlabel(
            'Electrolyzer price ($)',
        )
        plt.ylabel(
            r'$Hydrogen\ production\ cost\ (\$/kg)$'
        )
        plt.ylim([4.3,5.9])
        plt.legend(
            title = 'Operating condition'
        )
    
    def plot_4(self):
        # 不同电价与电流下，电解槽的最低制氢成本点变化
        electricity_price_range = np.arange(0.4,1.5,0.2)
        current_range = range(
            OperatingRange.Contour.Current.left,
            OperatingRange.Contour.Current.right,
            OperatingRange.Contour.Current.step
        )
        lye_temperature = 60
        lye_flow = OperatingCondition.Default.lye_flow
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        electricity_price = LifeCycle.electricity_price
        cooling_efficiency = LifeCycle.cooling_efficiency
        heating_efficiency = LifeCycle.heating_efficiency
        electrolyzer_price = 250000
        for electricity_price in electricity_price_range:
            cost_list = []
            for current in current_range:
                cost_cur = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current= current,
                    lye_temperature=lye_temperature,
                    lye_flow=lye_flow,
                    ambient_temperature=ambient_temperature,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                )
                cost_list.append(cost_cur)
            plt.plot(
                np.array(current_range) / self.electrolyzer.active_surface_area,
                cost_list,
                label = np.round(electricity_price*LifeCycle.RMB_2_USD,2)
            )
        plt.xlabel(
            r'$Current\ density (A/m^2)$'
        )
        plt.ylabel(
            r'$Hydrogen\ production\ cost\ (\$/kg)$'
        )
        plt.ylim([0,20])
        plt.legend(
            title = 'Electricity price ($/kWh)'
        )

    def plot_5(self):
        # 不同碱液温度与电流下，电解槽的最低制氢成本点变化
        lye_temperature_list = [40,50,60,70,80]
        current_range = range(
            OperatingRange.Contour.Current.left,
            OperatingRange.Contour.Current.right,
            OperatingRange.Contour.Current.step
        )
        
        lye_flow = OperatingCondition.Default.lye_flow
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        electricity_price = LifeCycle.electricity_price
        cooling_efficiency = LifeCycle.cooling_efficiency
        heating_efficiency = LifeCycle.heating_efficiency
        electrolyzer_price = 250000
        for lye_temperature in lye_temperature_list:
            cost_list = []
            for current in current_range:
                cost_cur = self.electrolyzer.hydrogen_cost_lifecycle_USD(
                    current= current,
                    lye_temperature=lye_temperature,
                    lye_flow=lye_flow,
                    ambient_temperature=ambient_temperature,
                    electricity_price=electricity_price,
                    electrolyzer_price=electrolyzer_price,
                    cooling_efficiency=cooling_efficiency,
                    heating_efficiency=heating_efficiency,
                )
                cost_list.append(cost_cur)
            plt.plot(
                np.array(current_range) / self.electrolyzer.active_surface_area,
                cost_list,
                label = np.round(lye_temperature,0)
            )
        plt.xlabel(
            r'$Current\ density (A/m^2)$'
        )
        plt.ylabel(
            r'$Hydrogen\ production\ cost\ (\$/kg)$'
        )
        plt.ylim([4.4,6.4])
        plt.legend(
            title = r'$Lye\ temperature\ (^\circ C)$'
        )

    def plot_6(self):
        electricity_price_range = np.arange(0.4,1.5,0.2)
        current_range = range(
            OperatingRange.Contour.Current.left,
            OperatingRange.Contour.Current.right,
            OperatingRange.Contour.Current.step
        )
        lye_temperature = 60
        lye_flow = OperatingCondition.Default.lye_flow
        ambient_temperature = OperatingCondition.Default.ambient_temperature
        electricity_price = LifeCycle.electricity_price
        cooling_efficiency = LifeCycle.cooling_efficiency
        heating_efficiency = LifeCycle.heating_efficiency
        electrolyzer_price = 250000
        electricity_price=0.4
        # for electricity_price in electricity_price_range:
        cost_list = []
        for current in current_range:
            cost_cur = self.electrolyzer.hydrogen_cost_lifecycle_detail_USD(
                current= current,
                lye_temperature=lye_temperature,
                lye_flow=lye_flow,
                ambient_temperature=ambient_temperature,
                electricity_price=electricity_price,
                electrolyzer_price=electrolyzer_price,
                cooling_efficiency=cooling_efficiency,
                heating_efficiency=heating_efficiency,
            )
            cost_list.append(cost_cur)
            if current == OperatingCondition.Default.current:
                cost_default = cost_cur
            if current == OperatingCondition.Optimal.current:
                cost_optimal = cost_cur
        cost_matrix = np.array(cost_list).T
        
        plt.stackplot(
            np.array(current_range) / self.electrolyzer.active_surface_area,
            cost_matrix,
            labels = (
                'Electrolyzer purchase',
                'Electrolysis cost',
                'Cooling and heating cost',
                'Additional cost',
            ),
            baseline='zero'
        )
        plt.xlabel(
            r'$Current\ density (A/m^2)$'
        )
        plt.ylabel(
            r'$Hydrogen\ production\ cost\ (\$/kg)$'
        )
        plt.ylim([0,6])
        plt.legend(
            title = 'Cost break down ($/kWh)'
        )
