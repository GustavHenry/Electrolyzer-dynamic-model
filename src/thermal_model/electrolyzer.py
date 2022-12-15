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

        self.interval = self.parameters.interval


    @staticmethod
    def voltage_thermal_neutral(temperature):
        """计算热中性电压, 根据出口温度
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
        """根据出口温度计算可逆电压，此方法针对的是单个数据，而非是data frame"""
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
        return self.polar_current_lh(
            current = current_density * self.active_surface_area,
            temperature=temperature
        )


    
    def faraday_efficiency_current(self, current, temperature):
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
    
    def power(self,current,voltage):
        return current * voltage / 1000 # kW

    def get_polarization(
        self,
        current_max,
        ambient_temperature,
        lye_flow,
        lye_temperature,
    ):
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



