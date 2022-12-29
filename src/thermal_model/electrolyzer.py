import numpy as np
import pandas as pd
from thermal_model.configs import *
from thermal_model.figure_plotter import *
from thermal_model.data import *
from utils_smooth import *
from loader import Loader
from plotter import Plotter
from utils_thermal_model_raw_process import *
from keys import *
import math
from thermal_model.configs import OperatingCondition


""" CLASS DEFINITION AND FUNCTIONS """


class Electrolyzer:
    """这里就是电解槽的类，所有和电解槽相关的内容都应该在这里面计算"""

    def __init__(self) -> None:
        self.default_ambient_temperature = OperatingCondition.Default.ambient_temperature   # degree C
        self.default_current = OperatingCondition.Default.current # A
        self.default_lye_flow = OperatingCondition.Default.lye_flow # m^3/h
        self.default_lye_temperature = OperatingCondition.Default.lye_temperature   # degree C

        self.parameters = ElectrolyzerParameter()
        self.num_cells = self.parameters.num_cells
        self.active_surface_area = self.parameters.active_surface_area
        self.coefficient_electric_heat = self.parameters.coefficient_electric_heat
        self.coefficient_radiation_dissipation = (
            self.parameters.coefficient_radiation_dissipation
        )
        self.coefficient_input_lye_heat = self.parameters.coefficient_input_lye_heat
        self.coefficient_output_lye_heat = self.parameters.coefficient_output_lye_heat
        self.coefficient_delta_temp = self.parameters.coefficient_delta_temp # 模型训练时的标的就是除过interval的
        self.correction_input_lye_heat = self.parameters.correction_input_lye_heat # 用于产生模型的热容

        self.heat_capacity_lye_flow = (
            self.coefficient_input_lye_heat / self.correction_input_lye_heat
        ) # 碱液热容
        self.interval = self.parameters.interval

    def current_2_density(self,current_range):
        # 将给定的电流转化为电流密度
        return np.array(
            current_range
        ) / self.active_surface_area


    @staticmethod
    def voltage_thermal_neutral(temperature):
        """计算热中性电压, 根据出口温度

        Args:
            temperature (_type_): 出口温度

        Returns:
            _type_: 小室的热中性电压
        """
        T_ref = 25
        F = 96485
        n = 2
        CH2O = 75  # 参考点状态下的水热容(单位：J/(K*mol))
        CH2 = 29
        CO2 = 29

        DHH2O = -2.86 * 10**5 + CH2O * (temperature - T_ref)  # 参考点状态下的焓变(单位：J/mol)
        DHH2 = 0 + CH2 * (temperature - T_ref)  # 参考点状态下的焓变(单位：J/mol)
        DHO2 = 0 + CO2 * (temperature - T_ref)  # 参考点状态下的焓变(单位：J/mol)
        voltage = (DHH2 + DHO2 / 2 - DHH2O) / (n * F)
        return voltage


    @staticmethod
    def voltage_reversible(temp_out):
        """根据出口温度计算可逆电压，此方法针对的是单个数据，而非是data frame

        Args:
            temp_out (_type_): 出口温度

        Returns:
            _type_: 小室的可逆电压
        """
        T_ref = 25
        F = 96485
        n = 2
        R = 8.3145
        CH2O = 75  # 参考点状态下的水热容(单位：J/(K*mol))
        CH2 = 29
        CO2 = 29
        S0_H2 = 131
        S0_H20 = 70
        S0_O2 = 205
        DHH2O = -2.86 * 10**5 + CH2O * (temp_out - T_ref)  # 参考点状态下的焓变(单位：J/mol)
        DHH2 = 0 + CH2 * (temp_out - T_ref)  # 参考点状态下的焓变(单位：J/mol)
        DHO2 = 0 + CO2 * (temp_out - T_ref)  # 参考点状态下的焓变(单位：J/mol)
        DH = DHH2 + DHO2 / 2 - DHH2O

        SH2 = (
            CH2 * np.math.log((temp_out + 273.15) / (T_ref + 273.15), 10)
            - R * np.math.log(10, 10)
            + S0_H2
        )
        SO2 = (
            CO2 * np.math.log((temp_out + 273.15) / (T_ref + 273.15), 10)
            - R * np.math.log(10, 10)
            + S0_O2
        )
        SH20 = CH2O * np.math.log((temp_out + 273.15) / (T_ref + 273.15), 10) + S0_H20
        DS = SH2 + 0.5 * SO2 - SH20
        DG = DH - (temp_out + 273.15) * DS

        return DG / (n * F)

    def thermal_balance_lsq(
        self,
        electric_heat,
        radiation_dissipation,
        input_lye_heat,
        output_lye_heat,
    ):
        """根据热模型，计算四个输入项得到的等式右端的温度变化量（每秒）

        Args:
            electric_heat (_type_): 电热，即电流乘电压
            radiation_dissipation (_type_): 辐射散失，即绝对温度的四次方减去环境绝对温度的四次方
            input_lye_heat (_type_): 输入热量，即碱液温度乘碱液流量
            output_lye_heat (_type_): 输出热量，即出口温度乘碱液流量

        Returns:
            _type_: 温度变化，每度/每秒
        """
        lsq_model_input = [
            electric_heat * self.coefficient_electric_heat,
            radiation_dissipation * self.coefficient_radiation_dissipation,
            input_lye_heat * self.coefficient_input_lye_heat,
            output_lye_heat * self.coefficient_output_lye_heat,
        ]
        return sum( lsq_model_input ) / self.coefficient_delta_temp


    def polar_current_lh(
        self,
        current,
        temperature,
    ):
        """李昊版本的极化曲线，只采用了出口温度作为指标

        Args:
            current (_type_): 输入电流，A
            temperature (_type_): 出口温度，摄氏度

        Returns:
            _type_: 整个电解槽的极化电压
        """
        current_density = current / self.active_surface_area
        cell_voltage = (
            Electrolyzer.voltage_reversible(
                temp_out=temperature
            ) + (
                (PolarizationLihao.r1 + PolarizationLihao.r2 * temperature)
                * current_density
            ) + (
                PolarizationLihao.s1
                + PolarizationLihao.s2 * temperature
                + PolarizationLihao.s3 * temperature**2
            ) * math.log(
                (
                    PolarizationLihao.t1
                    + PolarizationLihao.t2 / temperature
                    + PolarizationLihao.t3 / temperature**2
                ) * current_density + 1
            )
        )
        return cell_voltage * self.num_cells

    def polar_current_density_lh(self, current_density, temperature):
        """李昊版本的极化曲线的电流密度版本

        Args:
            current_density (_type_): 电流密度，A/m2
            temperature (_type_): 出口温度

        Returns:
            _type_: 电解槽的电压
        """
        return self.polar_current_lh(
            current = current_density * self.active_surface_area,
            temperature=temperature
        )


    
    def faraday_efficiency_current(self, current, temperature):
        """计算电解槽的法拉第效率

        Args:
            current (_type_): 电流
            temperature (_type_): 出口温度

        Returns:
            _type_: 法拉第效率，0-1之间
        """
        # NOTE: 原始数据拟合的时候，认为1700A对应2500A/m2的电流密度，所以认为活性面积为0.68
        # NOTE: 温度为出口温度
        current_density = current / 0.68      
        faraday_efficiency = (
            current_density ** 2 / (
                Faraday.f11 - Faraday.f12 * temperature + current_density ** 2
            )
        ) * ( Faraday.f21 + Faraday.f22 * temperature)
        return faraday_efficiency

    
    def faraday_efficiency_current_density(self,current_density,temperature):
        # NOTE: 原始数据拟合的时候，认为1700A对应2500A/m2的电流密度，所以认为活性面积为0.68，所以使用电流密度进行计算的时候需要进行矫正
        return self.faraday_efficiency_current(
            current=current_density * self.active_surface_area,
            temperature=temperature
        )


    def dT_current(
        self,
        temperature,
        ambient_temperature,
        lye_flow,
        lye_temperature,
        current,
    ):
        """计算当前状况下电解槽的温度变化情况，即每秒的温度变化

        Args:
            temperature (_type_): 出口温度
            ambient_temperature (_type_): 环境温度
            lye_flow (_type_): 碱液流量
            lye_temperature (_type_): 碱液入口温度
            current (_type_): 电流

        Returns:
            _type_: 电解槽出口温度变化，即每度/每秒
        """
        stack_voltage_thermal_neutral = Electrolyzer.voltage_thermal_neutral(
            temperature
        ) * self.num_cells   # V
        stack_voltage = self.polar_current_lh(current=current, temperature=temperature) # V

        electric_heat = ( stack_voltage - stack_voltage_thermal_neutral) * current / 1000   # kW
        radiation_dissipation = (
            temperature + Constants.absolute_temp_delta
        ) ** 4 - (
            ambient_temperature + Constants.absolute_temp_delta
        ) ** 4
        input_lye_heat = lye_flow * lye_temperature
        output_lye_heat = lye_flow * temperature
        return self.thermal_balance_lsq(
            electric_heat=electric_heat,
            radiation_dissipation=radiation_dissipation,
            input_lye_heat=input_lye_heat,
            output_lye_heat=output_lye_heat
        ) * self.interval

    
    def dT_current_density(
        self,
        temperature,
        ambient_temperature,
        lye_flow,
        lye_temperature,
        current_density,
    ):
        return self.dT_current(
            temperature=temperature,
            ambient_temperature=ambient_temperature,
            lye_flow=lye_flow,
            lye_temperature=lye_temperature,
            current = current_density *  self.active_surface_area
        )

    def dT_adiabatic_current(
        self,
        temperature,
        lye_flow,
        lye_temperature,
        current,
        
    ):
        """假设对电解槽进行绝热处理后，电解槽的温度变化情况，但是由于绝热处理产生的影响较小，所以可以忽略不计

        Args:
            temperature (_type_): _description_
            lye_flow (_type_): _description_
            lye_temperature (_type_): _description_
            current (_type_): _description_

        Returns:
            _type_: _description_
        """
        stack_voltage_thermal_neutral = Electrolyzer.voltage_thermal_neutral(temperature) * self.num_cells   # V
        stack_voltage = self.polar_current_lh(current=current, temperature=temperature) # V

        electric_heat = ( stack_voltage - stack_voltage_thermal_neutral) * current / 1000   # kW
        input_lye_heat = lye_flow * lye_temperature
        output_lye_heat = lye_flow * lye_temperature

        return self.thermal_balance_lsq(
            electric_heat=electric_heat,
            radiation_dissipation=0,
            input_lye_heat=input_lye_heat,
            output_lye_heat=output_lye_heat
        ) * self.interval

    def dT_adiabatic_current_density(
        self,
        temperature,
        lye_flow,
        lye_temperature,
        current_density,
        ambient_temperature = 15,
    ):
        return self.dT_adiabatic_current(
            temperature=temperature,
            lye_flow=lye_flow,
            lye_temperature=lye_temperature,
            current=current_density * self.active_surface_area,
            ambient_temperature = ambient_temperature
        )

    def temperature_thermal_balance_current(
        self,
        ambient_temperature,
        lye_flow,
        lye_temperature,
        current,
    ):
        """这里主要就是计算各种参数条件下，多少温度下会达到热平衡，也就是让最终的DeltaTemp为0"""
        """准备采用二分法计算最后的平衡温度大概是多少"""
        T_left = 0
        T_right = 200
        dT = 10
        while abs(dT) > dT_threshold:
            T_mid = (T_left + T_right) / 2.
            dT = self.dT_current(
                temperature=T_mid,
                ambient_temperature=ambient_temperature,
                lye_flow=lye_flow,
                lye_temperature=lye_temperature,
                current=current
            )
            if dT>0:
                T_left = T_mid
            else:
                T_right = T_mid
        return T_mid

    def temperature_thermal_balance_current_density(
        self,
        ambient_temperature,
        lye_flow,
        lye_temperature,
        current_density,
    ):
        return self.temperature_thermal_balance_current(
            ambient_temperature=ambient_temperature,
            lye_flow=lye_flow,
            lye_temperature=lye_temperature,
            current = current_density * self.active_surface_area
        )
    
    def temperature_thermal_adiabatic_current(
        self,
        lye_flow,
        lye_temperature,
        current
    ):
        """这里主要就是计算各种参数条件下，多少温度下会达到热平衡，也就是让最终的DeltaTemp为0"""
        """准备采用二分法计算最后的平衡温度大概是多少"""
        T_left = 0
        T_right = 200
        dT = 10
        while abs(dT) > dT_threshold:
            T_mid = (T_left + T_right) / 2.
            dT = self.dT_adiabatic_current(
                temperature=T_mid,
                lye_flow=lye_flow,
                lye_temperature=lye_temperature,
                current=current
            )
            if dT>0:
                T_left = T_mid
            else:
                T_right = T_mid
        return T_mid
    
    def temperature_thermal_adiabatic_current_density(
        self,
        lye_flow,
        lye_temperature,
        current_density,
    ):
        return self.temperature_thermal_adiabatic_current(
            lye_flow,
            lye_temperature,
            current_density * self.active_surface_area
        )

    def lye_temperature_for_given_condition_current(
        self,
        current,
        lye_flow,
        temperature,
        ambient_temperature,
    ):
        """在给定出口温度、碱液流量、电流的情况下求解此时的碱液温度
            因为没有发现更好的方法，暂时只能用dT来计算

        Args:
            lye_flow (_type_): 碱液流量
            temperature (_type_): 出口温度
            current (_type_): 电流
        """
        T_left = 0
        T_right = 150
        dT = 10
        while abs(dT)>1E-6:
            T_mid = (T_left + T_right) / 2
            dT = self.dT_current(
                current=current,
                temperature=temperature,
                ambient_temperature=ambient_temperature,
                lye_flow=lye_flow,
                lye_temperature=T_mid
            )
            if dT > 0:
                T_right = T_mid
            elif dT < 0 :
                T_left = T_mid
            else: 
                return T_mid
        return T_mid
    
    def lye_temperature_for_given_condition_current_density(
        self,
        current_density,
        lye_flow,
        temperature,
        ambient_temperature,
    ):
        return self.lye_temperature_for_given_condition_current(
            current_density * self.active_surface_area,
            lye_flow,
            temperature,
            ambient_temperature,
        )
 


    
    def power(self,current,voltage):
        """标准计算电解槽功率的方法

        Args:
            current (_type_): 电流
            voltage (_type_): 电解槽整体电压

        Returns:
            _type_: _description_
        """
        return current * voltage / 1000 # kW
    
    def cooling_power_requirement(
        self,
        temperature,
        lye_temperature,
        lye_flow
    ):
        """电解槽的冷却功率需求，乘上冷却系数就是真实的冷却功率

        Args:
            temperature (_type_): 出口温度
            lye_temperature (_type_): 碱液温度
            lye_flow (_type_): 碱液流量

        Returns:
            _type_: 冷却功率需求
        """
        return (
            temperature - lye_temperature
        ) * lye_flow * self.heat_capacity_lye_flow

    def power_total(
        self,
        current,
        ambient_temperature,
        lye_flow,
        lye_temperature,
        cooling_efficiency = LifeCycle.cooling_efficiency,
        heating_efficiency = LifeCycle.heating_efficiency
    ):
        """给定条件下给出电解槽系统的总功率

        Args:
            current (_type_): 电流，A
            ambient_temperature (_type_): 环境温度
            lye_flow (_type_): 碱液流量
            lye_temperature (_type_): 碱液温度

        Returns:
            _type_: 电解槽系统总功率，kW
        """
        temperature = self.temperature_thermal_balance_current(
            ambient_temperature=ambient_temperature,
            lye_flow=lye_flow,
            lye_temperature=lye_temperature,
            current=current,
        )
        cooling_power = self.cooling_power_requirement(
            temperature=temperature,
            lye_temperature=lye_temperature,
            lye_flow=lye_flow
        )

        voltage = self.polar_current_lh(
            current=current,
            temperature=temperature
        )
        power = self.power(
            current=current,
            voltage=voltage
        )
        if cooling_power>0:
            cost = power + cooling_power * cooling_efficiency
        else:
            cost = power + abs(cooling_power) * heating_efficiency
        return cost

    def power_total_current_density(
        self,
        current_density,
        lye_flow,
        lye_temperature,
        cooling_efficiency = LifeCycle.cooling_efficiency,
        heating_efficiency = LifeCycle.heating_efficiency
    ):
        return self.power_total(
            current=current_density * self.active_surface_area,
            lye_flow=lye_flow,
            lye_temperature=lye_temperature,
            cooling_efficiency=cooling_efficiency,
            heating_efficiency=heating_efficiency,
        )

    def efficiency_current(
        self,
        current,
        ambient_temperature,
        lye_flow,
        lye_temperature
    ):
        """计算给定条件下的高热值效率，即通过热中性电压除以当前电压再与法拉第效率乘积
        结果为百分比

        Args:
            current (_type_): 电流
            ambient_temperature (_type_): 环境温度
            lye_flow (_type_): 碱液流量
            lye_temperature (_type_): 碱液入口温度

        Returns:
            _type_: 高热值效率，百分比
        """
        #计算给定情况下的高热值效率
        temperature = self.temperature_thermal_balance_current(
            current = current,
            ambient_temperature=ambient_temperature,
            lye_flow=lye_flow,
            lye_temperature=lye_temperature
        )
        faraday_efficiency = self.faraday_efficiency_current(
            current=current,
            temperature=temperature
        )
        voltage = self.polar_current_lh(
            current=current,
            temperature=temperature
        )
        efficiency = self.voltage_thermal_neutral(
            temperature=temperature
        ) * self.num_cells / (
            voltage
        ) * faraday_efficiency * 100
        return efficiency

    def efficiency_current_density(
        self,
        current_density,
        ambient_temperature,
        lye_flow,
        lye_temperature
    ):
        return self.efficiency_current(
            self,
            current_density * self.active_surface_area,
            ambient_temperature,
            lye_flow,
            lye_temperature
        )

    def hydrogen_cost_current(
        self,
        current,
        ambient_temperature,
        lye_temperature,
        lye_flow,
        cooling_efficiency = LifeCycle.cooling_efficiency,
        heating_efficiency = LifeCycle.heating_efficiency
    ):
        """计算给定条件下的电解槽制氢功耗，计算中加入保持温度所需要的冷却功率和加热功率
        单位为kWh/Nm3

        Args:
            current (_type_): 电流
            ambient_temperature (_type_): 环境温度
            lye_temperature (_type_): 碱液温度
            lye_flow (_type_): 碱液流量
            cooling_efficiency (_type_, optional): 冷却效率，通常为0.35. Defaults to LifeCycle.cooling_efficiency.
            heating_efficiency (_type_, optional): 加热效率，通常为1. Defaults to LifeCycle.heating_efficiency.

        Returns:
            _type_: 制氢能耗，kWh/Nm3
        """
        power_total = self.power_total(
            current=current,
            ambient_temperature=ambient_temperature,
            lye_flow=lye_flow,
            lye_temperature=lye_temperature,
            cooling_efficiency=cooling_efficiency,
            heating_efficiency=heating_efficiency,
        )
        temperature = self.temperature_thermal_balance_current(
            ambient_temperature=ambient_temperature,
            lye_flow=lye_flow,
            lye_temperature=lye_temperature,
            current=current,
        )
        faraday_efficiency = self.faraday_efficiency_current(
            current=current,
            temperature=temperature
        )

        hydrogen_production_rate = (
            current / 2 / Constants.R
        ) * (
            Constants.std_volume /1000
        ) * self.num_cells * faraday_efficiency * Constants.seconds_in_hour
        return power_total / hydrogen_production_rate

    def hydrogen_cost_current_density(
        self,
        current_density,
        ambient_temperature,
        lye_temperature,
        lye_flow,
        cooling_efficiency = LifeCycle.cooling_efficiency,
        heating_efficiency = LifeCycle.heating_efficiency
    ):
        return self.hydrogen_cost_current(
            self,
            current_density * self.active_surface_area,
            ambient_temperature,
            lye_temperature,
            lye_flow,
            cooling_efficiency = cooling_efficiency,
            heating_efficiency = heating_efficiency
        )

    def hydrogen_cost_lifecycle(
        self,
        current,
        lye_temperature,
        lye_flow,
        ambient_temperature,
        electricity_price = LifeCycle.electricity_price,
        electrolyzer_price = LifeCycle.price,
        cooling_efficiency = LifeCycle.cooling_efficiency,
        heating_efficiency = LifeCycle.heating_efficiency,
        additional_cost = LifeCycle.additional_cost
    ):
        """计算给定条件下的全生命周期制氢能耗，RMB/kg H2

        Args:
            current (_type_): 电流
            lye_temperature (_type_): 碱液温度
            lye_flow (_type_): 碱液流量下
            ambient_temperature (_type_): 环境温度
            electricity_price (_type_, optional): 电价RMB/kWh. Defaults to LifeCycle.electricity_price.
            electrolyzer_price (_type_, optional): 电解槽采购价格RMB. Defaults to LifeCycle.price.
            cooling_efficiency (_type_, optional): 冷却效率，0~1. Defaults to LifeCycle.cooling_efficiency.
            heating_efficiency (_type_, optional): 加热效率，0~1. Defaults to LifeCycle.heating_efficiency.
            additional_cost (_type_, optional): 制氢的额外成本，0~1. Defaults to LifeCycle.additional_cost.

        Returns:
            _type_: _description_
        """
        electricity_lifecycle = self.power_total(
            current=current,
            lye_temperature=lye_temperature,
            lye_flow=lye_flow,
            ambient_temperature=ambient_temperature,
            cooling_efficiency=cooling_efficiency,
            heating_efficiency=heating_efficiency,
        ) * (
            LifeCycle.service_year * LifeCycle.hour_in_year * LifeCycle.service_rate
        )
        electricity_cost_lifecycle = electricity_lifecycle * electricity_price
        total_cost_lifecycle = electricity_cost_lifecycle + electrolyzer_price
        faraday_efficiency = self.faraday_efficiency_current(
            current=current,
            temperature=self.temperature_thermal_balance_current(
                current=current,
                ambient_temperature=ambient_temperature,
                lye_flow=lye_flow,
                lye_temperature=lye_temperature,
            )
        )
        hydrogen_production_hour = (
            current / Constants.weight_hydrogen / Constants.R 
        )* (
            Constants.std_volume / 1000
        ) * (
            self.num_cells * faraday_efficiency*Constants.seconds_in_hour
        ) * (
            Constants.weight_hydrogen/Constants.std_volume
        ) # 每小时氢气产量，kg
        hydrogen_production_year = hydrogen_production_hour * Constants.hours_in_year
        hydrogen_production_lifecycle = hydrogen_production_year * LifeCycle.service_year * LifeCycle.service_rate
        hydrogen_cost = total_cost_lifecycle / hydrogen_production_lifecycle * (
            1 + additional_cost
        )

        return hydrogen_cost
    
    def hydrogen_cost_lifecycle_current_density(
        self,
        current_density,
        lye_temperature,
        lye_flow,
        ambient_temperature,
        electricity_price = LifeCycle.electricity_price,
        electrolyzer_price = LifeCycle.price,
        cooling_efficiency = LifeCycle.cooling_efficiency,
        heating_efficiency = LifeCycle.heating_efficiency,
        additional_cost = LifeCycle.additional_cost
    ):
        return self.hydrogen_cost_lifecycle(
            current = current_density*self.active_surface_area,
            lye_temperature = lye_temperature,
            lye_flow = lye_flow,
            ambient_temperature = ambient_temperature,
            electricity_price = electricity_price,
            electrolyzer_price = electrolyzer_price,
            cooling_efficiency = cooling_efficiency,
            heating_efficiency = heating_efficiency,
            additional_cost = additional_cost
        )

    def get_polarization(
        self,
        current_max,
        ambient_temperature,
        lye_flow,
        lye_temperature,
    ):
        """获取给定限制下的电解槽极化特征

        Args:
            current_max (_type_): 最大电流
            ambient_temperature (_type_): 环境温度
            lye_flow (_type_): 碱液流量
            lye_temperature (_type_): 碱液入口温度

        Returns:
            _type_: 电流、电压、功率、出口温度数列
        """
        current_list = range(0,current_max,current_max//100)
        voltage_list = []
        power_list = []
        temperature_list = []

        for current in current_list:
            temperature = self.temperature_thermal_balance_current(
                ambient_temperature=ambient_temperature,
                lye_flow=lye_flow,
                lye_temperature=lye_temperature,
                current = current,
            )
            voltage = self.polar_current_lh(
                current=current,
                temperature=temperature,
            )
            voltage_list.append(voltage)
            temperature_list.append(temperature)
            power_list.append(self.power(current,voltage))

        return (
            current_list,
            voltage_list,
            power_list,
            temperature_list
        )

    def get_default_polarization(self):
        return self.get_polarization(
            current_max = 2000, # A
            ambient_temperature = self.default_ambient_temperature,
            lye_flow = self.default_lye_flow,
            lye_temperature = self.default_lye_temperature,
        )

    def show_polarization_curve(self):
        (
            current_list,
            voltage_list,
            power_list,
            temperature_list
        ) = self.get_default_polarization()
        figure = Model_default_polarization_curve(current_list=current_list,voltage_list=voltage_list)
        figure.save()
    
    def print_all_properties(self):
        keys = self.__dict__
        for k in keys:
            print(k , ':\t', keys[k])



