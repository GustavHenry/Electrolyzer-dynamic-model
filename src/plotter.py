import pandas as pd
import numpy as np
import os
from keys import *
import matplotlib.pyplot as plt
from aquarel import Theme
from abc import ABC, abstractmethod


class Plotter(ABC):
    """最基础的plotter类，对所有的plot做好基础调整"""

    def __init__(
        self,
        label="",
        title="",
        num_subplot=1,
        title_plot=True,
    ) -> None:
        self.label = label  # 以文章标题为单位的分类标准
        self.title = title  # 某文章内的图片标题
        self.num_subplot = num_subplot
        self.title_plot = title_plot  # 如果是直接放入文章的插图的话，就需要关闭
        self.dpi = 300
        self.figsize = {1: (8, 6), 2: (16, 6), 3: (18, 6), 4: (16, 12), 6: (16, 18)}
        self.figure = plt.figure(
            num=self.num_subplot, figsize=self.figsize[self.num_subplot], dpi=self.dpi
        )
        plt.subplots_adjust(
            left=0.18, bottom=0.1, right=0.87, top=0.9, wspace=0.15, hspace=0.15
        )
        # NOTE: 这里是个人定制的style，如果未来要修改，应当注意修改json中的font family，以满足中文显示的需求
        self.theme = Theme.from_file(Files.aquarel_theme_scientific)
        self.theme.apply()
        self.color_list = self.theme.params['colors']['palette']

    @abstractmethod
    def plot(self):
        """load and process required data from files"""
        raise NotImplementedError
    
    def plot_double_y_axis(
        self,
        x,
        y1,
        y2,
        x_title,
        y1_title,
        y2_title,
    ):
        plt.xlabel(x_title)
        ax1 = plt.gca()
        ax1.plot(
            x,
            y1,
            color = self.color_list[0]
        )
        ax1.tick_params(axis='y', colors=self.color_list[0])
        ax1.set_ylabel(y1_title)

        ax2 =ax1.twinx()
        ax2.plot(
            x,
            y2,
            color = self.color_list[1]
        )
        ax2.tick_params(axis='y', colors=self.color_list[1])
        ax2.set_ylabel(y2_title)
        
        plt.grid(visible=False)
        return ax1, ax2

    def plot_contour_map_with_2_points(
        self,
        matrix,
        x_range,
        y_range,
        value_default,
        value_optimal,
        xlabel = r'$Current\ density (A/m^2)$',
        ylabel = r'$Lye\ inlet\ temperature (^\circ C)$',
        unit = r'$^\circ C$'
    ):
        from thermal_model.configs import OperatingCondition
        temperature_maximum = max(matrix.flatten())
        temperature_minimum = min(matrix.flatten())
        contour_levels = np.arange(
            temperature_minimum,
            temperature_maximum,
            (temperature_maximum - temperature_minimum)/10
        )
        contour_figure = plt.contourf(
            np.array(x_range) / self.electrolyzer.active_surface_area,
            y_range,
            matrix,
            contour_levels,
            origin = 'upper'
        )
        plt.clabel(contour_figure, colors="w", fmt="%2.0f", fontsize=12)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
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
            str(np.round(value_default,1) )+ unit,
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
            str(np.round(value_optimal,1) )+ unit,
            color=OperatingCondition.Rated.color,
            fontdict={
                'size':PlotterOffset.Marker.Cross.Subplot4.font_size
            }
        ) 

    def save(self):
        if self.title_plot:
            plt.title(self.title)
        self.plot()

        if not os.path.exists(FIGURES_DIR):
            os.makedirs(FIGURES_DIR, exist_ok=True)
        self.file_name = " ".join(
            [
                self.label,
                self.title if not "$" in self.title else self.title.split("$")[0],
            ]
        )

        self.figure.savefig(os.path.join(FIGURES_DIR, self.file_name))
        print("figure saved at " + self.file_name)
        plt.show()


class QuadroPlotter(Plotter):
    def __init__(
        self, 
        label="",
        title="", 
        num_subplot=4, 
        title_plot=False
    ) -> None:
        super().__init__(label, title, num_subplot, title_plot)

    @abstractmethod
    def plot_1(self):
        raise NotImplementedError

    @abstractmethod
    def plot_2(self):
        raise NotImplementedError

    @abstractmethod
    def plot_3(self):
        raise NotImplementedError


    @abstractmethod
    def plot_4(self):
        raise NotImplementedError
    
    def plot(self):
        plt.subplots_adjust(
            left=0.05, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.15
        )
        plt.subplot(2,2,1)
        self.plot_1()
        plt.subplot(2,2,2)  
        self.plot_2()
        plt.subplot(2,2,3)
        self.plot_3()
        plt.subplot(2,2,4)
        self.plot_4()


# NOTE: 按照目前这个用法，其实可以取消这个类
class SinglePlotter(Plotter):
    def __init__(self, label="Thermal model", title="") -> None:
        super().__init__(num_subplot=1, label=label, title=title)
        self.label = label
        self.title = title
