#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
薛定谔方程 - 方势阱能级计算

本模块实现了一维方势阱中粒子能级的计算方法。
"""

import numpy as np
import matplotlib.pyplot as plt

# 物理常数
HBAR = 1.0545718e-34  # 约化普朗克常数 (J·s)
ELECTRON_MASS = 9.1094e-31  # 电子质量 (kg)
EV_TO_JOULE = 1.6021766208e-19  # 电子伏转换为焦耳的系数


def calculate_y_values(E_values, V, w, m):
    """
    计算方势阱能级方程中的三个函数值
    
    参数:
        E_values (numpy.ndarray): 能量值数组 (eV)
        V (float): 势阱高度 (eV)
        w (float): 势阱宽度 (m)
        m (float): 粒子质量 (kg)
    
    返回:
        tuple: 包含三个numpy数组 (y1, y2, y3)，分别对应三个函数在给定能量值下的函数值
    """
    # 转换能量单位为焦耳
    E_joules = E_values * EV_TO_JOULE
    V_joules = V * EV_TO_JOULE
    
    # 计算根号部分
    sqrt_part = np.sqrt(2 * m * E_joules) / HBAR
    
    # 计算 y1 = tan(w * sqrt(2*m*E)/(2*hbar))
    y1 = np.tan(w * sqrt_part / 2)
    
    # 计算 y2 = sqrt((V-E)/E)
    y2 = np.sqrt((V_joules - E_joules) / E_joules)
    
    # 计算 y3 = -sqrt(E/(V-E))
    y3 = -np.sqrt(E_joules / (V_joules - E_joules))
    
    # 处理接近V的能量值，避免除零错误
    mask_near_V = E_values > (V - 0.01)
    y2[mask_near_V] = np.nan
    y3[mask_near_V] = np.nan
    
    return y1, y2, y3


def plot_energy_functions(E_values, y1, y2, y3):
    """
    绘制能级方程的三个函数曲线
    
    参数:
        E_values (numpy.ndarray): 能量值数组 (eV)
        y1 (numpy.ndarray): 函数y1的值
        y2 (numpy.ndarray): 函数y2的值
        y3 (numpy.ndarray): 函数y3的值
    
    返回:
        matplotlib.figure.Figure: 绘制的图形对象
    """
    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制 y1 曲线（蓝色实线）
    ax.plot(E_values, y1, 'b-', label=r'$\tan\left(\frac{w\sqrt{2mE}}{2\hbar}\right)$', alpha=0.7)
    
    # 绘制 y2 曲线（红色虚线，偶宇称）
    ax.plot(E_values, y2, 'r--', label=r'$\sqrt{\frac{V-E}{E}}$ (偶宇称)', alpha=0.7)
    
    # 绘制 y3 曲线（绿色点线，奇宇称）
    ax.plot(E_values, y3, 'g-.', label=r'$-\sqrt{\frac{E}{V-E}}$ (奇宇称)', alpha=0.7)
    
    # 设置坐标轴范围
    ax.set_xlim(0, 20)
    ax.set_ylim(-10, 10)
    
    # 添加标题和标签
    ax.set_title('一维方势阱能级方程函数图', fontsize=16)
    ax.set_xlabel('能量 (eV)', fontsize=14)
    ax.set_ylabel('函数值', fontsize=14)
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # 添加水平和垂直参考线
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    for n in range(6):
        ax.axvline(x=n*3.5, color='gray', linestyle=':', alpha=0.2)
    
    return fig


def find_energy_level_bisection(n, V, w, m, precision=0.001, E_min=0.001, E_max=None):
    """
    使用二分法求解方势阱中的第n个能级
    
    参数:
        n (int): 能级序号 (0表示基态，1表示第一激发态，以此类推)
        V (float): 势阱高度 (eV)
        w (float): 势阱宽度 (m)
        m (float): 粒子质量 (kg)
        precision (float): 求解精度 (eV)
        E_min (float): 能量搜索下限 (eV)
        E_max (float): 能量搜索上限 (eV)，默认为V
    
    返回:
        float: 第n个能级的能量值 (eV)
    """
    # 设置默认的E_max
    if E_max is None:
        E_max = V - 0.001  # 确保E_max小于V
    
    # 确定能级的奇偶性
    is_even = (n % 2 == 0)
    
    # 调整搜索区间，根据能级序号大致估计可能的位置
    E_min = max(E_min, 0.001)
    E_max = min(E_max, V - 0.001)
    
    # 二分法迭代
    while (E_max - E_min) > precision:
        E_mid = (E_min + E_max) / 2
        
        # 计算函数值
        y1, y2, y3 = calculate_y_values(np.array([E_mid]), V, w, m)
        
        # 根据奇偶性选择对应的函数
        if is_even:
            # 偶宇称：y1 = y2
            f_mid = y1[0] - y2[0]
        else:
            # 奇宇称：y1 = y3
            f_mid = y1[0] - y3[0]
        
        # 调整搜索区间
        if f_mid > 0:
            E_max = E_mid
        else:
            E_min = E_mid
    
    # 返回中点作为结果
    return (E_min + E_max) / 2


def main():
    """
    主函数，执行方势阱能级的计算和可视化
    """
    # 参数设置
    V = 20.0  # 势阱高度 (eV)
    w = 1e-9  # 势阱宽度 (m)
    m = ELECTRON_MASS  # 粒子质量 (kg)
    
    # 1. 计算并绘制函数曲线
    E_values = np.linspace(0.001, 19.999, 5000)  # 增加点数以获得更平滑的曲线
    y1, y2, y3 = calculate_y_values(E_values, V, w, m)
    fig = plot_energy_functions(E_values, y1, y2, y3)
    plt.savefig('energy_functions.png', dpi=300)
    plt.show()
    
    # 2. 使用二分法计算前6个能级
    energy_levels = []
    for n in range(6):
        # 为每个能级调整搜索范围，提高二分法效率
        E_min = 0.001 + n * 3.0
        E_max = min(19.999, E_min + 3.0)
        energy = find_energy_level_bisection(n, V, w, m, E_min=E_min, E_max=E_max)
        energy_levels.append(energy)
        print(f"能级 {n}: {energy:.3f} eV")
    
    # 与参考值比较
    reference_levels = [0.318, 1.270, 2.851, 5.050, 7.850, 11.215]
    print("\n参考能级值:")
    for n, ref in enumerate(reference_levels):
        print(f"能级 {n}: {ref:.3f} eV")
    
    # 计算并显示误差
    print("\n误差分析:")
    for n, (calc, ref) in enumerate(zip(energy_levels, reference_levels)):
        error = abs(calc - ref) / ref * 100
        print(f"能级 {n}: 计算值 = {calc:.3f} eV, 参考值 = {ref:.3f} eV, 误差 = {error:.2f}%")


if __name__ == "__main__":
    main()
