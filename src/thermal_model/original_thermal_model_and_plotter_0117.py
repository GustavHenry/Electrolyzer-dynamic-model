"""EALIER FUNCTIONS"""
import time
import pywt  # python wavelet transmission
import numpy as np

def WL(data_input, threshold=0.3):  # 小波分解
    index = []
    data = []
    for i in range(len(data_input) - 1):
        X = float(i)
        Y = float(data_input[i])
        index.append(X)
        data.append(Y)
    # 创建小波对象并定义参数:
    w = pywt.Wavelet("db8")  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    coeffs = pywt.wavedec(data, "db8", level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
    data_output = pywt.waverec(coeffs, "db8")
    if len(data_output) != len(data_input):
        data_output = np.append(data_output, data_output[len(data_output) - 1])
    return data_output


""" FUNCTIONS USED IN THE CLASS"""


def Vtn(Temp):
    """这里就是计算热中性电压"""
    T_ref = 25
    F = 96485
    n = 2
    CH2O = 75  # 参考点状态下的水热容(单位：J/(K*mol))
    CH2 = 29
    CO2 = 29
    DHH2O = -2.86 * 10**5 + CH2O * (Temp - T_ref)  # 参考点状态下的焓变(单位：J/mol)
    DHH2 = 0 + CH2 * (Temp - T_ref)  # 参考点状态下的焓变(单位：J/mol)
    DHO2 = 0 + CO2 * (Temp - T_ref)  # 参考点状态下的焓变(单位：J/mol)
    return (DHH2 + DHO2 / 2 - DHH2O) / (n * F)


def Vres(Temp):
    """这里就是计算热可逆电压"""
    

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
    DHH2O = -2.86 * 10**5 + CH2O * (Temp - T_ref)  # 参考点状态下的焓变(单位：J/mol)
    DHH2 = 0 + CH2 * (Temp - T_ref)  # 参考点状态下的焓变(单位：J/mol)
    DHO2 = 0 + CO2 * (Temp - T_ref)  # 参考点状态下的焓变(单位：J/mol)
    DH = DHH2 + DHO2 / 2 - DHH2O

    SH2 = (
        CH2 * np.math.log((Temp + 273.15) / (T_ref + 273.15), 10)
        - R * np.math.log(10, 10)
        + S0_H2
    )
    SO2 = (
        CO2 * np.math.log((Temp + 273.15) / (T_ref + 273.15), 10)
        - R * np.math.log(10, 10)
        + S0_O2
    )
    SH20 = CH2O * np.math.log((Temp + 273.15) / (T_ref + 273.15), 10) + S0_H20
    DS = SH2 + 0.5 * SO2 - SH20
    DG = DH - (Temp + 273.15) * DS

    return DG / (n * F)


def LyeHeatUP(x0=165, k=0.03, high=46, ambT=16):


    x = np.arange(3 * x0)
    Tlye = high / (1 + np.exp(-k * (x - x0))) + ambT
    return Tlye


def LyeCoolDOWN(timespan=1100, x0=60, ambT=21, high=39, k=0.007):


    x = np.arange(timespan)
    y = high * np.exp(-k * (x - x0)) + ambT
    for i in range(x0):
        y[i] = high + ambT
    return y


""" CLASS DEFINITION AND FUNCTIONS """


class Electrolyzer:
    """这里就是电解槽的类，所有和电解槽相关的内容都应该在这里面计算"""

    ambT_default = 25
    current_default = 1500
    T_default = 82
    Qlye_default = 1.2
    Tlye_default = 60

    def __init__(
        self,
        r_active=368,
        r_margin=53,
        r_endplate=524,
        thickness_plate=10.4,
        thickness_electrode=1,
        thickness_insulator=3.75,
        thickness_endplate=97,
        Power_rated=125,
        current_density=4000,
        interval=20,
    ):
        """输入时的长度单位应当都为毫米 mm，功率单位应当为千瓦 kW，电流单位为安培每平方米 A/m2"""
        """
        r_active = 465.2,#活性部分的半径
        r_margin =100,#活性部分外围边框的半径
        r_endplate=524,#段板直径
        thickness_plate=10.4,#极板厚度
        thickness_electrode = 1.5,#活性部分的电极厚度（用来计算碱液量）
        thickness_insulator=3.75,#隔膜或者垫片厚度，待定
        thickness_endplate = 97, #端板厚度
        Power = 125,#额定功率，kW
        current_density = 2500#电流密度
        interval = 20 #s, the sampling interval of the model
        """
        r_plate = r_active + r_margin
        self.surf_area_active = 3.1415926 * r_active**2 / 1000**2  # square meters
        self.n_cell = round(
            Power_rated * 1000 / 2.15 / (current_density * self.surf_area_active)
        )  # integer, 34 is default value
        self.total_length = (
            (self.n_cell + 1) * thickness_plate
            + thickness_insulator * self.n_cell
            + thickness_endplate * 2
        ) / 1000  # meters
        self.total_space = (
            (self.n_cell + 1) * 3.1415926 * ((r_plate) ** 2) * thickness_plate
            + self.n_cell * 3.1415926 * ((r_plate) ** 2) * thickness_insulator
            + 3.1415926 * thickness_endplate * r_endplate**2
        ) / 1000**3  # cubic meters
        self.vacancy_space = (
            self.n_cell
            * 3.1415926
            * r_active**2
            * (thickness_plate - thickness_electrode)
        ) / 1000**3  # cubic meters, room for alkaline
        self.occupied_space = (
            self.total_space
            - self.vacancy_space
            - self.n_cell * 3.1415926 * (r_plate) ** 2 * thickness_insulator / 1000**3
        )  # cubic meters room for ss304
        self.rate_vacancy = self.vacancy_space / self.total_space  # vacancy rate
        self.surface_area = (
            ((self.n_cell + 1) * thickness_plate + self.n_cell * thickness_insulator)
            * 3.1415926
            * 2
            * (r_plate)
            + 4 * 3.1415926 * r_endplate**2
            - 2 * 3.1415926 * (r_plate) ** 2
            + 2 * 3.1415926 * 2 * r_endplate * thickness_endplate
        ) / 1000**2  # square meters, for radiation calculation
        self.interval = interval  # s, the sampling interval of the model
        self.max_current_density = current_density

    def merge_coef(
        self,
        theta_power=0,
        epsilon_electrolyzer=0.95,
        rho_steel=7900,
        rho_alkaline=1280,
        HC_steel=0.50,
        HC_alkaline=2.2625,
    ):
        """最后输入到模型的只有四个输入，发热功率，辐射功率，碱液热输入功率，碱液热输出功率，对标的就是温度变化乘热容"""
        C_0 = 5.67  # W/(m2K4)黑体的辐射常数
        """
        rho_steel = 7900 #kg/m3, material used as the structural part， ss304
        rho_alkaline = 1280 #kg/m3, alkaline
        epsilon_electrolyzer = 0.95 #电解槽表面的系统发射率
        theta_power = 0 #coef of heat power, [0,1)
        HC_steel = 0.500 #kJ/(kg*K)
        HC_alkaline = 2.2625 #kJ/(kg*K)
        ####correction of coefficients
        corr_power = 1
        corr_radiation = 1
        corr_HeatLyeIn = 1
        corr_HeatLyeOut = 1
        corr_DeltaTemp = 1
        """
        self.epsilon = epsilon_electrolyzer
        self.weight_structural = self.occupied_space * rho_steel  # kg
        self.weight_alkaline_inside = self.vacancy_space * rho_alkaline  # kg
        self.weight_total = self.weight_structural + self.weight_alkaline_inside  # kg
        self.HC_structural = self.weight_structural * HC_steel
        self.HC_alkaline_inside = self.weight_alkaline_inside * HC_alkaline
        self.HC_total = self.HC_structural + self.HC_alkaline_inside
        self.HC_alkaline_in_flux = (
            1000.0 / 3600.0 * rho_alkaline * HC_alkaline / 1000.0
        )  # kJ/s, can be directly timed with flux of alkaline and alkaline inlet temperature
        """ deriving coefficients of the model"""
        """归一化操作可能也得在这里进行，处以一个系数"""
        # a = [9.04111877e-04, -9.47759032e-13, 6.14396832e-04, -6.28809630e-04]#目标模型的参数 1029
        # a = [ 5.26170236e-04, -1.22881062e-12,  6.60662596e-04, -5.95708584e-04]#1001
        a = [
            9.76913600e-04,
            -3.19807956e-13,
            5.66544937e-04,
            -6.41418079e-04,
        ]  # 目标模型的参数 1014
        self.Model_Target = [1, a[1] / a[0], a[2] / a[0], a[3] / a[0], 1 / a[0]]
        if 0:  # 这一部分针对的如果模型需要重新进行参数矫正的话，才需要启用这一模块
            self.coef_power = corr_power * 1.0 / (1.0 - theta_power)
            self.coef_radiation = (
                corr_radiation
                * epsilon_electrolyzer
                * C_0
                * self.surface_area
                / 1000
                / 100**4
            )  # kJ/s
            self.coef_HeatLyeIn = (
                corr_HeatLyeIn * self.HC_alkaline_in_flux
            )  # kJ/s, can be directly timed with flux of alkaline and alkaline inlet temperature
            self.coef_HeatLyeOut = (
                corr_HeatLyeOut * self.HC_alkaline_in_flux
            )  # kJ/s, can be directly timed with flux of alkaline and alkaline inlet temperature
            self.coef_DeltaTemp = corr_DeltaTemp * self.HC_total  # kJ/s
            """ correction coefficients"""
            self.corr_power = self.Model_Target[0] / self.coef_power
            self.corr_radiation = self.Model_Target[1] / self.coef_radiation
            self.corr_HeatLyeIn = self.Model_Target[2] / self.coef_HeatLyeIn
            self.corr_HeatLyeOut = self.Model_Target[3] / self.coef_HeatLyeOut
            self.corr_DeltaTemp = self.Model_Target[4] / self.coef_DeltaTemp
            """ corrected coefficients of model"""
            self.coef_power = self.corr_power * 1.0 / (1.0 - theta_power)
            self.coef_radiation = (
                self.corr_radiation
                * epsilon_electrolyzer
                * C_0
                * self.surface_area
                / 1000
                / 100**4
            )  # kJ/s
            self.coef_HeatLyeIn = (
                self.corr_HeatLyeIn * self.HC_alkaline_in_flux
            )  # kJ/s, can be directly timed with flux of alkaline and alkaline inlet temperature
            self.coef_HeatLyeOut = (
                self.corr_HeatLyeOut * self.HC_alkaline_in_flux
            )  # kJ/s, can be directly timed with flux of alkaline and alkaline inlet temperature
            self.coef_DeltaTemp = self.corr_DeltaTemp * self.HC_total  # kJ/s
        if 1:  # 这里固化一下校正系数，正常情况下直接调用即可
            self.corr_power = 1.0
            self.corr_radiation = -1.4214310326772457
            self.corr_HeatLyeIn = 0.7209118268943362
            self.corr_HeatLyeOut = -0.8161857055567431
            self.corr_DeltaTemp = 1.0362887600924242

            self.coef_power = self.corr_power * 1.0 / (1.0 - theta_power)
            self.coef_radiation = (
                self.corr_radiation
                * epsilon_electrolyzer
                * C_0
                * self.surface_area
                / 1000
                / 100**4
            )  # kJ/s
            self.coef_HeatLyeIn = (
                self.corr_HeatLyeIn * self.HC_alkaline_in_flux
            )  # kJ/s, can be directly timed with flux of alkaline and alkaline inlet temperature
            self.coef_HeatLyeOut = (
                self.corr_HeatLyeOut * self.HC_alkaline_in_flux
            )  # kJ/s, can be directly timed with flux of alkaline and alkaline inlet temperature
            self.coef_DeltaTemp = self.corr_DeltaTemp * self.HC_total  # kJ/s

    """这里只能计算单点的状态"""

    def polar(self, current=current_default, Temp=T_default):
        """polarization curve"""
        import math

        r1 = 0.0001362
        r2 = -1.316e-06
        s1 = 0.06494
        s2 = 0.0013154
        s3 = -4.296e-06
        t1 = 0.1645
        t2 = -18.96
        t3 = 672.5
        j = current / self.surf_area_active
        U = (
            Vres(Temp)
            + (r1 + r2 * Temp) * j
            + (s1 + s2 * Temp + s3 * Temp**2)
            * math.log((t1 + t2 / Temp + t3 / Temp**2) * j + 1)
        )
        return U * self.n_cell

    """这里只能计算单点的状态"""

    def Farady(self, current=current_default, Temp=T_default):
        true_current_density = current / self.surf_area_active  # 这个是最后需要返回到外面的电流
        fake_current_density = current / 0.68  # 标准计算实在2500电流密度下进行的，所以这里的活性面积是0.68
        f11 = 1.067e4
        f12 = 101.1
        f21 = 0.989
        f22 = 7.641e-5
        F_efficiency = (
            fake_current_density**2 / (f11 - f12 * Temp + fake_current_density**2)
        ) * (f21 + f22 * Temp)
        return F_efficiency, true_current_density

    def dT_calculation(
        self,
        T,
        ambT=ambT_default,
        Qlye=Qlye_default,
        Tlye=Tlye_default,
        current=current_default,
    ):
        voltage_thermal_neutral = Vtn(T) * self.n_cell
        voltage = self.polar(current, Temp=T)
        """这里所有的值，应该都和模型对应起来，不应当直接具有物理意义，应当是最后再给物理意义"""
        E = (voltage - voltage_thermal_neutral) * current / 1000  # kW
        """这里有个注意事项，本来应该除以100的四次方，我们把这个内容交给最后的模型系数去做，所以才会让模型系数非常小
        也测试了如果把这些0放在原始数据处理之后，系数的变化也只是这几个0，没有其他影响"""
        Rad = (T + 273.15) ** 4 - (ambT + 273.15) ** 4
        HeatLyeIn = Qlye * Tlye
        HeatLyeIOut = Qlye * T
        Model_input = [
            E * self.coef_power,
            Rad * self.coef_radiation,
            HeatLyeIn * self.coef_HeatLyeIn,
            HeatLyeIOut * self.coef_HeatLyeOut,
        ]
        dT = sum(Model_input) / self.coef_DeltaTemp  # 这里就是要计算当前条件下的温度的变化
        return dT * self.interval

    def dT_calculation_adiabatic(
        self,
        T,
        ambT=ambT_default,
        Qlye=Qlye_default,
        Tlye=Tlye_default,
        current=current_default,
    ):
        """这里主要是考虑将电解槽设置为绝热状态，也就是不考虑辐射散热"""
        voltage_thermal_neutral = Vtn(T) * self.n_cell
        voltage = self.polar(current, Temp=T)
        """这里所有的值，应该都和模型对应起来，不应当直接具有物理意义，应当是最后再给物理意义"""
        E = (voltage - voltage_thermal_neutral) * current / 1000  # kW
        HeatLyeIn = Qlye * Tlye
        HeatLyeIOut = Qlye * T
        Model_input = [
            E * self.coef_power,
            HeatLyeIn * self.coef_HeatLyeIn,
            HeatLyeIOut * self.coef_HeatLyeOut,
        ]
        dT = sum(Model_input) / self.coef_DeltaTemp  # 这里就是要计算当前条件下的温度的变化
        return dT * self.interval

    def T_thermal_eq(
        self,
        ambT: object = ambT_default,
        Qlye: object = Qlye_default,
        Tlye: object = Tlye_default,
        current: object = current_default,
    ) -> object:
        """这里主要就是计算各种参数条件下，多少温度下会达到热平衡，也就是让最终的DeltaTemp为0"""
        """准备采用二分法计算最后的平衡温度大概是多少"""
        T_left = 0
        T_right = 200
        dT = 10
        while abs(dT) > 1e-8:
            T_mid = (T_right + T_left) / 2
            dT = self.dT_calculation(T_mid, ambT, Qlye, Tlye, current)
            if dT > 0:
                T_left = T_mid
            if dT < 0:
                T_right = T_mid
        return T_mid

    def T_thermal_eq_adiabatic(
        self,
        ambT: object = ambT_default,
        Qlye: object = Qlye_default,
        Tlye: object = Tlye_default,
        current: object = current_default,
    ) -> object:
        """和上面的区别，计算一下绝热状态下的电解槽平衡温度"""
        """这里主要就是计算各种参数条件下，多少温度下会达到热平衡，也就是让最终的DeltaTemp为0"""
        """准备采用二分法计算最后的平衡温度大概是多少"""
        T_left = 0
        T_right = 200
        dT = 10
        while abs(dT) > 1e-8:
            T_mid = (T_right + T_left) / 2
            dT = self.dT_calculation_adiabatic(T_mid, ambT, Qlye, Tlye, current)
            if dT > 0:
                T_left = T_mid
            if dT < 0:
                T_right = T_mid
        return T_mid

    def temp_flow(self, current_seq, Tlye_seq, T0=60, Qlye=Qlye_default):
        dT_seq = [0]
        T_seq = [T0]
        for i, n in enumerate(current_seq):
            v = self.polar(Temp=T_seq[-1], current=n)
            dT = self.dT_calculation(
                T=T_seq[-1], ambT=25, Qlye=Qlye, Tlye=Tlye_seq[i], current=n
            )
            dT_seq.append(dT)
            T_seq.append(T_seq[-1] + dT)
        return T_seq, dT_seq

    def polarizationcurve(
        self, current_max=2000, ambT=ambT_default, Qlye=Qlye_default, Tlye=Tlye_default
    ):
        import matplotlib.pyplot as plt

        current = range(0, current_max, 50)
        self.polar_current = []
        self.polar_volatege = []
        self.polar_power = []
        self.polar_temp = []
        for c in current:
            j = c / self.surf_area_active
            t = self.T_thermal_eq(ambT=ambT, Qlye=Qlye, Tlye=Tlye, current=c)
            v = self.polar(current=c, Temp=t)
            self.polar_temp.append(t)
            self.polar_current.append(j)
            self.polar_volatege.append(v)
            self.polar_power.append(c * v / 1000)
        return (
            self.polar_current,
            self.polar_volatege,
            self.polar_power,
            self.polar_temp,
        )

    def Print_All_Properties(self):
        keys = self.__dict__
        for k in keys:
            print(k, "||", keys[k])

    def Print_Model_coefs(self):
        coefs = [
            self.coef_power,
            self.coef_radiation,
            self.coef_HeatLyeIn,
            self.coef_HeatLyeOut,
            self.coef_DeltaTemp,
        ]
        print("target ", self.Model_Target)
        print("model ", coefs)


"""MAIN"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time


def Polar_Curve_Tlye(current_max=2000):
    # 不同碱液温度下电解槽的出口温度
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-colorblind")
    plt.figure(figsize=(10, 6))
    ele = Electrolyzer()
    ele.merge_coef()
    temp_seq = np.arange(15, 95, 10)

    for t in temp_seq:
        i, v, p, t_eq = ele.polarizationcurve(current_max=current_max, Tlye=t)
        plt.plot(i, v, linestyle="-")
        # plt.plot(i,t_eq,linestyle = '--')
    plt.xticks()
    plt.yticks()
    plt.xlabel(r"$Current\ density\ (A/m^2)$")
    # plt.ylabel(r'$Electrolyzer\ voltage (V)$')
    plt.ylabel(r"$Electrolyzer\ outlet temperature\ (V)$")
    plt.legend(
        [
            r"$35\circ C$",
            r"$45\circ C$",
            r"$55\circ C$",
            r"$65\circ C$",
            r"$75\circ C$",
            r"$85\circ C$",
        ],
        loc=7,
    )
    # plt.title('Electrolyzer polarization curve under different lye input temperatures')
    plt.title(
        "Electrolyzer outlet temperature under different current densities and lye input temperatures"
    )
    plt.grid()
    plt.show()


def current_temp_farady():
    """这个函数主要是计算不同温度、电流密度下的法拉地效率"""
    plt.figure(figsize=(10, 6))
    ele = Electrolyzer()
    ele.merge_coef()
    cur_seq = range(2000)
    T_seq = np.arange(35, 105, 20)
    for t in T_seq:
        cd_seq = []
        FE_seq = []
        for c in cur_seq:
            FE, cd = ele.Farady(current=c, Temp=t)
            cd_seq.append(cd)
            FE_seq.append(FE)
        plt.plot(cd_seq, FE_seq)
    print(T_seq)
    plt.xticks()
    plt.yticks()
    plt.xlabel(r"$Current\ density\ (A/m^2)$")
    # plt.ylabel(r'$Electrolyzer\ voltage (V)$')
    plt.ylabel(r"$Electrolyzer\ farady \ efficiency$")
    plt.legend(
        [r"$35^\circ C$", r"$55^\circ C$", r"$75^\circ C$", r"$95^\circ C$"], loc=7
    )
    # plt.title('Electrolyzer polarization curve under different lye input temperatures')
    plt.grid()
    plt.show()


def current_TlyeIn_TlyeOut():
    """这里是画不同碱液入口温度、输入电流下的碱液出口平衡温度"""
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    Tlye_seq = np.arange(35, 95, 1)
    I_seq = np.arange(100, 2050, 25)
    ambt = 45
    # 0,15,30,45
    res_mat = np.ones((len(Tlye_seq), len(I_seq)))

    for i in range(len(I_seq)):
        for j in range(len(Tlye_seq)):
            # res_mat[j,i] = ele.T_thermal_eq(current=I_seq[i],Tlye=Tlye_seq[j],ambT=ambt)#这里就是画不同的工况下的平衡温度
            res_mat[j, i] = (
                ele.T_thermal_eq(current=I_seq[i], Tlye=Tlye_seq[j], ambT=ambt)
                - Tlye_seq[j]
            )  # 这里就是画不同的工况下的平衡温度与入口温度的差值
            # res_mat[j, i] = ele.T_thermal_eq(current=I_seq[i], Tlye=Tlye_seq[j], ambT=45) -  ele.T_thermal_eq(current=I_seq[i],Tlye=Tlye_seq[j],ambT=0)#这一部分是画不同环境温度下电解草出口温度的差值
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.074, bottom=0.095, right=1, top=0.967)
    """这一部分主要就是各种画图的设置"""
    min_res = min(res_mat.flatten())
    max_res = max(res_mat.flatten())
    levels = np.arange(min_res, max_res, (max_res - min_res) / 20)
    CS = plt.contourf(
        I_seq / ele.surf_area_active,
        Tlye_seq,
        res_mat,
        levels,
        origin="upper",
        cmap="viridis",
    )
    # colormap: 'inferno','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_
    plt.colorbar(CS)
    plt.xlabel(r"$current \ density \ (A/m^2)$")
    plt.ylabel(r"$Electrolyzer \ inlet \ temperature \ (^\circ C)$")
    plt.clabel(CS, colors="w", fmt="%2.1f", fontsize=12, manual=False)
    print("the consumed time is", time.time() - t0)
    print(
        "the consumed time per point is",
        (time.time() - t0) / (len(I_seq) * len(Tlye_seq)),
    )
    plt.show()


def current_TlyeIn_TlyeOut_adiabatic():
    """这里是画不同碱液入口温度、输入电流下的碱液出口平衡温度"""
    """这一部分主要考虑的是绝热状态下的结果"""
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    Qlye = 1.2
    Tlye_seq = np.arange(35, 95, 1)
    I_seq = np.arange(100, 2050, 25)
    ambt = 15
    # 0,15,30,45
    res_mat = np.ones((len(Tlye_seq), len(I_seq)))

    for i in range(len(I_seq)):
        for j in range(len(Tlye_seq)):
            # res_mat[j,i] = ele.T_thermal_eq_adiabatic(current=I_seq[i],Tlye=Tlye_seq[j],ambT=ambt)#这里就是画不同的工况下的平衡温度，这里和环境温度没什么关系
            # res_mat[j, i] = ele.T_thermal_eq_adiabatic(current=I_seq[i],Tlye=Tlye_seq[j],ambT=ambt) \
            #                - ele.T_thermal_eq(current=I_seq[i], Tlye=Tlye_seq[j], ambT=ambt)  # 这里就是画不同的工况下的平衡温度 与15摄氏度环境温度下的差值
            res_mat[j, i] = (
                -(
                    ele.T_thermal_eq_adiabatic(current=I_seq[i], Tlye=Tlye_seq[j])
                    * ele.coef_HeatLyeOut
                    + Tlye_seq[j] * ele.coef_HeatLyeIn
                )
                * Qlye
            )  # 这里是绝热状态下冷却功率
            res_mat[j, i] += (
                ele.T_thermal_eq(current=I_seq[i], Tlye=Tlye_seq[j], ambT=ambt)
                * ele.coef_HeatLyeOut
                + Tlye_seq[j] * ele.coef_HeatLyeIn
            ) * Qlye  # 绝热下冷却与15摄氏度冷却的差值
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.074, bottom=0.095, right=1, top=0.967)
    """这一部分主要就是各种画图的设置"""
    min_res = min(res_mat.flatten())
    max_res = max(res_mat.flatten())
    levels = np.arange(min_res, max_res, (max_res - min_res) / 20)
    CS = plt.contourf(
        I_seq / ele.surf_area_active,
        Tlye_seq,
        res_mat,
        levels,
        origin="upper",
        cmap="viridis",
    )
    # colormap: 'inferno','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_
    plt.colorbar(CS)
    plt.xlabel(r"$current \ density \ (A/m^2)$")
    plt.ylabel(r"$Electrolyzer \ inlet \ temperature \ (^\circ C)$")
    plt.clabel(CS, colors="w", fmt="%2.1f", fontsize=12, manual=True)
    print("the consumed time is", time.time() - t0)
    print(
        "the consumed time per point is",
        (time.time() - t0) / (len(I_seq) * len(Tlye_seq)),
    )
    plt.show()


def Power_eq():
    """这里是计算碱液出口与入口温度差为零时，不同入口温度下的电流密度"""
    ele = Electrolyzer()
    ele.merge_coef()
    Tlye_seq = np.arange(35, 95, 1)
    res = np.ones(len(Tlye_seq))
    ambt = range(0, 50, 15)
    # 0,15,30,45
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.074, bottom=0.095, right=0.964, top=0.967)
    for a in ambt:
        for j in range(len(Tlye_seq)):
            I_left = 0
            I_right = 1500
            dT = 100
            while abs(dT) > 1e-2:
                I_mid = (I_left + I_right) / 2
                T_eq = ele.T_thermal_eq(current=I_mid, Tlye=Tlye_seq[j], ambT=a)
                dT = T_eq - Tlye_seq[j]
                if dT > 0:
                    I_right = I_mid
                elif dT < 0:
                    I_left = I_mid
            res[j] = (
                I_mid * ele.polar(I_mid, Temp=Tlye_seq[j]) / 1000
            )  # 进入口温度一样的时候，电解槽需要的输入功率
        plt.plot(Tlye_seq, res)
    plt.grid()
    plt.legend(["Amb Temp = 0", "Amb Temp = 15", "Amb Temp = 30", "Amb Temp = 45"])
    plt.xlabel(r"$Electrolyzer\ input\ temperature\ (^\circ C)$")
    plt.ylabel(r"$Electrolyzer \ input \ power \ (kW)$")
    plt.show()


def highest_lye_out():
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    Tlye = 85
    I = 1800

    ambT = np.arange(0, 50)
    Highest_lye_temp = []
    t = Tlye
    i = I
    for at in ambT:
        Highest_lye_temp.append(ele.T_thermal_eq(current=i, ambT=at, Tlye=t))
    plt.plot(Highest_lye_temp)
    plt.show()


def current_TlyeIn_cooling():
    """这一部分主要是针对碱液出口与入口温度差以及冷却功率的买普图"""
    """冷却干脆不算矫正系数了，直接就碱液流量乘上温度差什么的"""
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    ambt = 45
    # 0,15,30,45
    Tlye_seq = np.arange(35, 95, 2)
    I_seq = np.arange(100, 2050, 50)
    Qlye_input = 1.2
    res_mat = np.ones((len(Tlye_seq), len(I_seq)))
    for i in range(len(I_seq)):
        for j in range(len(Tlye_seq)):
            # res_mat[j,i] = ele.T_thermal_eq(current=I_seq[i],Tlye=Tlye_seq[j]) - Tlye_seq[j]#这一部分是画碱液进出口温度差的
            res_mat[j, i] = (
                (
                    ele.T_thermal_eq(current=I_seq[i], Tlye=Tlye_seq[j], ambT=ambt)
                    - Tlye_seq[j]
                )
                * (ele.coef_HeatLyeIn / ele.corr_HeatLyeIn)
                * Qlye_input
            )

    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.074, bottom=0.05, right=1, top=0.967)
    """这一部分主要就是各种画图的设置"""
    min_res = min(res_mat.flatten())
    max_res = max(res_mat.flatten())
    levels = np.arange(min_res, max_res, (max_res - min_res) / 20)
    for i, n in enumerate(levels):
        levels[i] = int(n)
    CS = plt.contourf(
        I_seq / ele.surf_area_active,
        Tlye_seq,
        res_mat,
        levels,
        origin="upper",
        cmap="viridis",
    )
    # colormap: 'inferno','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_
    plt.colorbar(CS)
    plt.xlabel(r"$current \ density \ (A/m^2)$")
    plt.ylabel(r"$Electrolyzer \ inlet \ temperature \ (^\circ C)$")
    plt.clabel(CS, colors="w", fmt="%2.0f", fontsize=12, manual=True)
    print("the consumed time is", time.time() - t0)
    print(
        "the consumed time per point is",
        (time.time() - t0) / (len(I_seq) * len(Tlye_seq)),
    )
    plt.show()


def cooling_Power_Qlye():
    """这里是画不同碱液流量下的冷却功率需求，是线图"""
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    ambt = 15
    Qlye_seq = np.arange(0, 2.6, 0.1)
    Tlye_input = 60
    I_seq = np.arange(500, 4500, 500)
    # I_seq.reverse()
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.074, bottom=0.093, right=0.964, top=0.967)
    for i in I_seq:
        res = []
        i = i * ele.surf_area_active  # 换算一下电流密度
        for q in Qlye_seq:
            cooling_power = (
                (
                    ele.T_thermal_eq(current=i, Tlye=Tlye_input, ambT=ambt, Qlye=q)
                    - Tlye_input
                )
                * (ele.coef_HeatLyeIn / ele.corr_HeatLyeIn)
                * q
                / 125
                * 100
            )  # 冷却功率需求
            T_eq = ele.T_thermal_eq(
                current=i, Tlye=Tlye_input, ambT=ambt, Qlye=q
            )  # 平衡温度
            res.append(T_eq)
        plt.plot(Qlye_seq, res)
    plt.grid()
    plt.legend(I_seq)
    # plt.legend(['current input = 1100 A','current input = 1300 A','current input = 1500 A','current input = 1700 A'])
    plt.xlabel(r"$Flow\ of\ lye\ (m^3/h)$")
    plt.ylabel(r"$Electrolyzer \ thermal \ balance \ temperature \ ( ^\circ C)$")
    # plt.ylabel(r'$Cooling \ power\ requirement \ of \ rated \ power \ ( \%)$')
    print("the consumed time is", time.time() - t0)
    plt.show()


def current_TlyeIn_Energy_cost_efficiency():
    """这里主要是各种环境温度与工况下的电解槽高热值效率"""
    plt.figure(figsize=(10, 6))
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    ambt = 30
    Tlye_seq = np.arange(35, 95, 2)
    I_seq = np.arange(100, 2050, 50)
    res_mat = np.ones((len(Tlye_seq), len(I_seq)))
    for i in range(len(I_seq)):
        for j in range(len(Tlye_seq)):
            T_outlet_cur = ele.T_thermal_eq(
                Tlye=Tlye_seq[j], current=I_seq[i], ambT=ambt
            )
            FE, cd = ele.Farady(current=I_seq[i], Temp=T_outlet_cur)
            res_mat[j, i] = (
                Vtn(T_outlet_cur)
                * ele.n_cell
                / ele.polar(current=I_seq[i], Temp=T_outlet_cur)
                * FE
            )
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.064, bottom=0.085, right=1, top=0.967)
    """这一部分主要就是各种画图的设置"""
    min_res = min(res_mat.flatten())
    max_res = max(res_mat.flatten())
    levels = np.arange(min_res, max_res, (max_res - min_res) / 20)
    CS = plt.contourf(
        I_seq / ele.surf_area_active,
        Tlye_seq,
        res_mat,
        levels,
        origin="upper",
        cmap="viridis",
    )
    # colormap: 'inferno','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_
    plt.colorbar(CS)
    plt.xlabel(r"$current \ density \ (A/m^2)$")
    plt.ylabel(r"$Electrolyzer \ inlet \ temperature \ (^\circ C)$")
    plt.clabel(CS, colors="w", fmt="%1.3f", fontsize=12, manual=True)
    print("the consumed time is", time.time() - t0)
    print(
        "the consumed time per point is",
        (time.time() - t0) / (len(I_seq) * len(Tlye_seq)),
    )
    plt.show()


def current_TlyeIn_overalcost():
    """这部分应该是电解水制氢的总体氢耗"""
    """这里调整了字体大小，可以作为参考模板"""
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    Tlye_seq = np.arange(35, 95, 2)
    I_seq = np.arange(100, 2050, 10)
    res_mat = np.ones((len(Tlye_seq), len(I_seq)))
    Qlye = 1
    cooling_efficiency = 0.5  # 这个是每冷却1kw所需要的能量
    fonts = 13  # 字体大小
    for i in range(len(I_seq)):
        for j in range(len(Tlye_seq)):
            T_eq = ele.T_thermal_eq(current=I_seq[i], Tlye=Tlye_seq[j], Qlye=Qlye)
            cooling_power = (
                (T_eq - Tlye_seq[j]) * Qlye * (ele.coef_HeatLyeIn / ele.corr_HeatLyeIn)
            )
            ele_power = ele.polar(current=I_seq[i], Temp=T_eq) * I_seq[i] / 1000
            overall_power = abs(cooling_power) * cooling_efficiency + ele_power
            FE, cd = ele.Farady(current=I_seq[i], Temp=T_eq)
            Q_H2 = (I_seq[i] / 2 / 96485) * (22.4 / 1000) * ele.n_cell * FE * 3600
            res_mat[j, i] = overall_power / Q_H2
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.06, bottom=0.09, right=1, top=0.967)
    """这一部分主要就是各种画图的设置"""
    min_res = min(res_mat.flatten())
    max_res = max(res_mat.flatten())
    max_res = 5.7
    levels = np.arange(min_res, max_res, (max_res - min_res) / 20)
    # levels = np.array([4.2,4.5,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,6.8,7.5])
    CS = plt.contourf(
        I_seq / ele.surf_area_active,
        Tlye_seq,
        res_mat,
        levels,
        origin="upper",
        cmap="viridis",
    )
    # colormap: 'inferno','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_
    plt.colorbar(CS)
    plt.xlabel(r"$current \ density \ (A/m^2)$", fontsize=fonts)
    plt.ylabel(r"$Electrolyzer \ inlet \ temperature \ (^\circ C)$", fontsize=fonts)
    plt.clabel(CS, colors="w", fmt="%1.2f", manual=True, fontsize=fonts)
    print("the consumed time is", time.time() - t0)
    print(
        "the consumed time per point is",
        (time.time() - t0) / (len(I_seq) * len(Tlye_seq)),
    )
    plt.show()


def current_TlyeIn_liftcycle_cost():
    """这部分应该是电解水制氢的全生命周期氢耗"""
    """这里调整了字体大小，可以作为参考模板"""
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    Tlye_seq = np.arange(35, 95, 2)
    I_seq = np.arange(100, 2050, 10)
    res_mat = np.ones((len(Tlye_seq), len(I_seq)))
    Qlye = 1
    cooling_efficiency = 0.5  # 这个是每冷却1kw所需要的能量
    fonts = 13  # 字体大小
    Price = 300000  # 整套装置的原始价格为76.2万元人民币
    Years = 15  # 计划服役年限为15年
    Hours = 24 * 365  # 每年有这么多个小时
    Open = 0.8  # 开工率为70%
    Electricity = 0.4  # 假设电价为0.4元人民币每度电
    Earning = 0.2  # 利润率为20%
    for i in range(len(I_seq)):
        for j in range(len(Tlye_seq)):
            ii = I_seq[i]
            tlye = Tlye_seq[j]
            T_eq = ele.T_thermal_eq(current=ii, Tlye=tlye, Qlye=Qlye)
            cooling_power = (
                (T_eq - tlye) * Qlye * (ele.coef_HeatLyeIn / ele.corr_HeatLyeIn)
            )
            ele_power = ele.polar(current=ii, Temp=T_eq) * I_seq[i] / 1000
            overall_power = abs(cooling_power) * cooling_efficiency + ele_power
            overall_power *= Years * Hours * Open
            Cost_energy = overall_power * Electricity
            FE, cd = ele.Farady(current=ii, Temp=T_eq)
            Q_H2 = (ii / 2 / 96485) * (22.4 / 1000) * ele.n_cell * FE * 3600
            Q_H2 *= Years * Hours * Open * 0.089
            res_mat[j, i] = (Cost_energy + Price) / Q_H2 * (1 + Earning)
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.06, bottom=0.09, right=1, top=0.967)
    """这一部分主要就是各种画图的设置"""
    min_res = min(res_mat.flatten())
    max_res = max(res_mat.flatten())
    max_res = 50
    levels = np.arange(min_res, max_res, (max_res - min_res) / 35)
    # levels = np.array([4.2,4.5,4.8,5.0,5.2,5.4,5.6,5.8,6.0,6.2,6.4,6.6,6.8,7.5])
    CS = plt.contourf(
        I_seq / ele.surf_area_active,
        Tlye_seq,
        res_mat,
        levels,
        origin="upper",
        cmap="viridis",
    )
    # colormap: 'inferno','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_
    plt.colorbar(CS)
    plt.xlabel(r"$current \ density \ (A/m^2)$", fontsize=fonts)
    plt.ylabel(r"$Electrolyzer \ inlet \ temperature \ (^\circ C)$", fontsize=fonts)
    plt.clabel(CS, colors="w", fmt="%1.2f", manual=True, fontsize=fonts)
    print("the consumed time is", time.time() - t0)
    print(
        "the consumed time per point is",
        (time.time() - t0) / (len(I_seq) * len(Tlye_seq)),
    )
    plt.show()


# current_TlyeIn_liftcycle_cost()


def Energy_flow_ele_heat():
    """这里主要是计算不同电流密度、碱液入口温度之下的电解槽发热功率"""
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    ambT = 45
    # 0,15,30,45
    Tlye_seq = np.arange(35, 95, 2)
    I_seq = np.arange(100, 2050, 10)
    res_mat = np.ones((len(Tlye_seq), len(I_seq)))
    for i in range(len(I_seq)):
        for j in range(len(Tlye_seq)):
            T_eq = ele.T_thermal_eq(current=I_seq[i], Tlye=Tlye_seq[j], ambT=ambT)
            V_tn = Vtn(Temp=T_eq) * ele.n_cell
            V_ele = ele.polar(current=I_seq[i], Temp=T_eq)
            res_mat[j, i] = (V_ele - V_tn) * I_seq[i] * ele.coef_power / 1000
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.064, bottom=0.095, right=1, top=0.967)
    """这一部分主要就是各种画图的设置"""
    min_res = min(res_mat.flatten())
    max_res = max(res_mat.flatten())
    levels = np.arange(min_res, max_res, (max_res - min_res) / 20)
    CS = plt.contourf(
        I_seq / ele.surf_area_active,
        Tlye_seq,
        res_mat,
        levels,
        origin="upper",
        cmap="viridis",
    )
    # colormap: 'inferno','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_
    plt.colorbar(CS)
    plt.xlabel(r"$current \ density \ (A/m^2)$")
    plt.ylabel(r"$Electrolyzer \ inlet \ temperature \ (^\circ C)$")
    plt.clabel(CS, colors="w", fmt="%1.1f", fontsize=12, manual=True)
    print("the consumed time is", time.time() - t0)
    print(
        "the consumed time per point is",
        (time.time() - t0) / (len(I_seq) * len(Tlye_seq)),
    )
    plt.show()


def Energy_flow_rad_heat():
    """这里主要是计算不同电流密度、碱液入口温度之下的辐射散热功率"""
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    ambT = 45
    # 0,15,30,45
    Tlye_seq = np.arange(35, 95, 2)
    I_seq = np.arange(100, 2050, 10)
    res_mat = np.ones((len(Tlye_seq), len(I_seq)))
    for i in range(len(I_seq)):
        for j in range(len(Tlye_seq)):
            T_eq = ele.T_thermal_eq(current=I_seq[i], Tlye=Tlye_seq[j], ambT=ambT)
            res_mat[j, i] = ele.coef_radiation * (
                (T_eq + 273.15) ** 4 - (ambT + 273.15) ** 4
            )
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.064, bottom=0.095, right=1, top=0.967)
    """这一部分主要就是各种画图的设置"""
    min_res = min(res_mat.flatten())
    max_res = max(res_mat.flatten())
    levels = np.arange(min_res, max_res, (max_res - min_res) / 20)
    CS = plt.contourf(
        I_seq / ele.surf_area_active,
        Tlye_seq,
        res_mat,
        levels,
        origin="upper",
        cmap="viridis",
    )
    # colormap: 'inferno','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_
    plt.colorbar(CS)
    plt.xlabel(r"$current \ density \ (A/m^2)$")
    plt.ylabel(r"$Electrolyzer \ inlet \ temperature \ (^\circ C)$")
    plt.clabel(CS, colors="w", fmt="%1.1f", fontsize=12, manual=True)
    print("the consumed time is", time.time() - t0)
    print(
        "the consumed time per point is",
        (time.time() - t0) / (len(I_seq) * len(Tlye_seq)),
    )
    plt.show()


def Energy_flow_delta_lye():
    """这里主要是计算不同电流密度、碱液入口温度之下的辐射散热功率"""
    t0 = time.time()
    ele = Electrolyzer()
    ele.merge_coef()
    ambT = 45
    # 0,15,30,45
    Qlye = 1.2
    Tlye_seq = np.arange(35, 95, 2)
    I_seq = np.arange(100, 2050, 10)
    res_mat = np.ones((len(Tlye_seq), len(I_seq)))
    for i in range(len(I_seq)):
        for j in range(len(Tlye_seq)):
            Heat_lye_in = ele.coef_HeatLyeIn * Qlye * Tlye_seq[j]
            T_eq = ele.T_thermal_eq(current=I_seq[i], Tlye=Tlye_seq[j], ambT=ambT)
            Heat_lye_out = ele.coef_HeatLyeOut * Qlye * T_eq
            res_mat[j, i] = Heat_lye_in + Heat_lye_out
    plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.064, bottom=0.095, right=1, top=0.967)
    """这一部分主要就是各种画图的设置"""
    min_res = min(res_mat.flatten())
    max_res = max(res_mat.flatten())
    levels = np.arange(min_res, max_res, (max_res - min_res) / 20)
    CS = plt.contourf(
        I_seq / ele.surf_area_active,
        Tlye_seq,
        res_mat,
        levels,
        origin="upper",
        cmap="viridis",
    )
    # colormap: 'inferno','Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_
    plt.colorbar(CS)
    plt.xlabel(r"$current \ density \ (A/m^2)$")
    plt.ylabel(r"$Electrolyzer \ inlet \ temperature \ (^\circ C)$")
    plt.clabel(CS, colors="w", fmt="%1.1f", fontsize=12, manual=True)
    print("the consumed time is", time.time() - t0)
    print(
        "the consumed time per point is",
        (time.time() - t0) / (len(I_seq) * len(Tlye_seq)),
    )
    plt.show()
