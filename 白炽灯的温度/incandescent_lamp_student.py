#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白炽灯温度优化 - 学生代码模板

请根据项目说明实现以下函数，完成白炽灯效率与最优温度的计算。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar

# 物理常数
H = 6.62607015e-34  # 普朗克常数 (J·s)
C = 299792458       # 光速 (m/s)
K_B = 1.380649e-23  # 玻尔兹曼常数 (J/K)

# 可见光波长范围 (m)
VISIBLE_LIGHT_MIN = 390e-9  # 390 nm
VISIBLE_LIGHT_MAX = 750e-9  # 750 nm


def planck_law(wavelength, temperature):
    """
    计算普朗克黑体辐射公式
    
    参数:
        wavelength (float or numpy.ndarray): 波长，单位为米
        temperature (float): 温度，单位为开尔文
    
    返回:
        float or numpy.ndarray: 给定波长和温度下的辐射强度 (W/(m²·m·sr))
    """
    # 避免除零错误
    if isinstance(wavelength, np.ndarray):
        wavelength = np.where(wavelength == 0, 1e-10, wavelength)
    else:
        wavelength = wavelength if wavelength != 0 else 1e-10
        
    # 普朗克定律公式
    term1 = (2 * H * C**2) / (wavelength**5)
    term2 = np.exp((H * C) / (wavelength * K_B * temperature)) - 1
    intensity = term1 / term2
    
    return intensity


def calculate_visible_power_ratio(temperature):
    """
    计算给定温度下可见光功率与总辐射功率的比值
    
    参数:
        temperature (float): 温度，单位为开尔文
    
    返回:
        float: 可见光效率（可见光功率/总功率）
    """
    # 直接积分普朗克定律
    def integrand(wavelength):
        return planck_law(wavelength, temperature)  # 修正：移除了错误的*wavelength因子
    
    # 计算可见光波段的积分
    visible_power, _ = integrate.quad(integrand, VISIBLE_LIGHT_MIN, VISIBLE_LIGHT_MAX)
    
    # 计算总辐射功率（扩展积分范围到10000nm以覆盖更多红外辐射）
    total_power, _ = integrate.quad(integrand, 1e-9, 10000e-9)  # 修正：积分上限从1000nm提高到10000nm
    
    # 计算效率
    efficiency = visible_power / total_power
    
    return efficiency


def plot_efficiency_vs_temperature(temp_range):
    """
    绘制效率-温度关系曲线
    
    参数:
        temp_range (numpy.ndarray): 温度范围，单位为开尔文
    
    返回:
        tuple: (matplotlib.figure.Figure, numpy.ndarray, numpy.ndarray) 图形对象、温度数组、效率数组
    """
    # 计算每个温度点的效率
    efficiencies = np.array([calculate_visible_power_ratio(temp) for temp in temp_range])
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(temp_range, efficiencies, 'b-', linewidth=2)
    
    # 设置图表属性
    ax.set_xlabel('Temperature (K)', fontsize=12)
    ax.set_ylabel('Visible Light Efficiency', fontsize=12)
    ax.set_title('Incandescent Lamp Efficiency vs Temperature', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return fig, temp_range, efficiencies


def find_optimal_temperature():
    """
    寻找使白炽灯效率最大的最优温度
    
    返回:
        tuple: (float, float) 最优温度和对应的效率
    """
    # 定义目标函数（取负值以便使用最小化算法）
    def objective_function(temperature):
        return -calculate_visible_power_ratio(temperature)
    
    # 使用黄金分割法寻找最优温度
    result = minimize_scalar(
        objective_function,
        bounds=(1000, 10000),
        method='bounded',
        options={'xatol': 1.0}  # 精度要求为1K
    )
    
    optimal_temp = result.x
    optimal_efficiency = -result.fun
    
    return optimal_temp, optimal_efficiency


def main():
    """
    主函数，计算并可视化最优温度
    """
    # 绘制效率-温度曲线 (1000K-10000K)
    temp_range = np.linspace(1000, 10000, 100)
    fig_efficiency, temps, effs = plot_efficiency_vs_temperature(temp_range)
    plt.savefig('efficiency_vs_temperature.png', dpi=300)
    plt.show()
    
    # 计算最优温度
    optimal_temp, optimal_efficiency = find_optimal_temperature()
    print(f"\n最优温度: {optimal_temp:.1f} K")
    print(f"最大效率: {optimal_efficiency:.4f} ({optimal_efficiency*100:.2f}%)")
    
    # 与实际白炽灯温度比较
    actual_temp = 2700
    actual_efficiency = calculate_visible_power_ratio(actual_temp)
    print(f"\n实际灯丝温度: {actual_temp} K")
    print(f"实际效率: {actual_efficiency:.4f} ({actual_efficiency*100:.2f}%)")
    print(f"效率差异: {(optimal_efficiency - actual_efficiency)*100:.2f}%")
    
    # 标记最优和实际温度点
    plt.figure(figsize=(10, 6))
    plt.plot(temps, effs, 'b-')
    plt.plot(optimal_temp, optimal_efficiency, 'ro', markersize=8, label=f'Optimal: {optimal_temp:.1f} K')
    plt.plot(actual_temp, actual_efficiency, 'go', markersize=8, label=f'Actual: {actual_temp} K')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Visible Light Efficiency')
    plt.title('Incandescent Lamp Efficiency vs Temperature')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('optimal_temperature.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
