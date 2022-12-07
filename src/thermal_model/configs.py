from keys import Cols
import numpy as np
from keys import Constants

thermal_model_input_cols = [
    Cols.electric_heat,
    Cols.radiation_dissipation,
    Cols.input_lye_heat,
    Cols.output_lye_heat,
]

thermal_model_output_cols = [Cols.delta_temp]

dT_threshold = 1e-8 # 计算热平衡温度时的温度差值阈值


class PolarizationLihao:
    r1 = 0.0001362
    r2 = -1.316e-06
    s1 = 0.06494
    s2 = 0.0013154
    s3 = -4.296e-06
    t1 = 0.1645
    t2 = -18.96
    t3 = 672.5


class ElectrolyzerParameter:
    def __init__(self) -> None:
        
        self.radius_active = 367.8  # mm
        self.radius_margin = 53  # mm
        self.radius_endplate = 524  # mm

        self.thickness_plate = 10.4  # mm
        self.thickness_electrode = 1  # mm
        self.thickness_insulator = 3.75  # mm
        self.thickness_endplate = 97  # mm

        self.power_rated = 125  # kWh
        self.current_density_max = 4000  # A/m^2
        self.interval = 20  # s

        self.theta_power = 0  # [0,1), coefficient of heat power,

        self.coefficient_lsq = [
            9.76913600e-04,
            -3.19807956e-13,
            5.66544937e-04,
            -6.41418079e-04,
        ]  # 目标模型的参数 1014
        # NOTE: 分别对应着 electric_heat, radiation_dissipation, input_lye_heat, output_lye_heat
        # NOTE: 因为是线性模型的结果参数，所以说上面的参数都是直接应用在输入项之前的，而温度变化的系数应当是1

        self.coefficient_electric_heat = 1
        self.coefficient_radiation_dissipation = self.coefficient_lsq[1] / self.coefficient_lsq[0]
        self.coefficient_input_lye_heat = self.coefficient_lsq[2] / self.coefficient_lsq[0]
        self.coefficient_output_lye_heat = self.coefficient_lsq[3] / self.coefficient_lsq[0]
        self.coefficient_delta_temp = 1 / self.coefficient_lsq[0]

        # NOTE: 下面的矫正系数是指经过变换之后的线性方程的校正系数结果，即各项输入项之前都已经乘上了物理参数
        self.correction_electric_heat = 1.0
        self.correction_radiation_dissipation = -1.4214310326772457
        self.correction_input_lye_heat = 0.7209118268943362
        self.correction_output_lye_heat = -0.8161857055567431
        self.correction_delta_temp = 1.0362887600924242

        self.radius_plate =  self.radius_active + self.radius_margin  # mm

        self.active_surface_area =  np.pi * self.radius_active**2 / (1000**2)  # m^2

        self.num_cells =  34  # int, number of cells, default as 34

        self.length_total =  (
            (self.num_cells + 1) * self.thickness_plate
            + self.num_cells * self.thickness_insulator
            + self.thickness_endplate * 2
        ) / 1000  # m

        self.space_total =  (
            (self.num_cells + 1) * np.pi * (self.radius_plate**2) * self.thickness_plate
            + self.num_cells * np.pi * (self.radius_plate**2) * self.thickness_insulator
            + np.pi * self.thickness_endplate * self.radius_endplate**2
        ) / (
            1000**3
        )  # m^3

        self.space_vacancy =  (
            self.num_cells
            * np.pi
            * self.radius_active**2
            * (self.thickness_plate - self.thickness_electrode)
        ) / (
            1000**3
        )  # m^3

        self.space_vacancy_rate =  self.space_vacancy / self.space_total

        self.space_occupied =  self.space_total - self.space_vacancy

        self.surface_area =  (
            (
                (self.num_cells + 1) * self.thickness_plate
                + self.num_cells * self.thickness_insulator
            )
            * np.pi
            * 2
            * self.radius_plate
            + (
                4 * np.pi * self.radius_endplate**2
                - 2 * np.pi * self.radius_plate
                + 2 * np.pi * 2 * self.radius_endplate * self.thickness_endplate
            )
        ) / (
            1000**2
        )  # m^2

        self.weight_structural =  self.space_occupied * Constants.rho_steel  # kg

        self.weight_lye_inside =  self.space_vacancy * Constants.rho_alkaline  # kg

        self.weight_total =  self.weight_structural + self.weight_lye_inside  # kg

        self.heat_capacity_structural =  self.weight_structural * Constants.rho_steel  # kJ/K

        self.heat_capacity_lye_inside =  self.weight_lye_inside * Constants.rho_alkaline  # kJ/K

        self.heat_capacity_total =  self.heat_capacity_structural + self.heat_capacity_lye_inside  # kJ/K

        self.heat_capacity_rate_lye =(
            Constants.rho_alkaline * Constants.specific_heat_capacity_alkaline / 3600.0
        )  # kJ/(m^3 K)
        #heat capacity rate of the lye flow, can be directly timed with lye flow and temperature
        # which will become the heat rate


    def coefficients_physical(self):
        # NOTE: 这里指的是在线性方程中，假设可以直接成立的情况下的物理方程中的参数，即功率的项前系数为1的原始物理方程
        self.coefficient_physical_electric_power = 1 / (1 - self.theta_power)
        self.coefficient_physical_radiation_dissipation = (
            Constants.epsilon_steel * Constants.C_0 * self.surface_area
        ) / (
            1000 * 100**4
        )  # kW
        self.coefficient_physical_input_lye_heat = (
            self.heat_capacity_rate_lye
        )  # kJ/(m^3 K)
        self.coefficient_physical_output_lye_heat = (
            self.heat_capacity_rate_lye
        )  # kJ/(m^3 K)
        self.coefficient_physical_delta_temp = self.heat_capacity_total  # kJ/K
