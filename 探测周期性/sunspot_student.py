#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 学生代码模板

请根据项目说明实现以下函数，完成太阳黑子效率与最优温度的计算。
"""

import numpy as np
import matplotlib.pyplot as plt

def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据
    
    参数:
        url (str): 本地文件路径
        
    返回:
        tuple: (years, sunspots) 年份和太阳黑子数
    """
    data = np.loadtxt(url)
    years = data[:, 1]
    sunspots = data[:, 3]
    # 处理缺失值（以-1表示）
    mask = sunspots >= 0
    years = years[mask]
    sunspots = sunspots[mask]
    return years, sunspots

def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子数据随时间变化图
    
    参数:
        years (numpy.ndarray): 年份数组
        sunspots (numpy.ndarray): 太阳黑子数数组
    """
    plt.figure(figsize=(12, 6))
    plt.plot(years, sunspots)
    plt.title('太阳黑子数随时间变化')
    plt.xlabel('年份')
    plt.ylabel('太阳黑子数')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 估计周期（目视）
    estimated_period = float(input("请根据图像估计周期（月）："))
    print(f"目视估计周期：{estimated_period} 月")

def compute_power_spectrum(sunspots):
    """
    计算太阳黑子数据的功率谱
    
    参数:
        sunspots (numpy.ndarray): 太阳黑子数数组
        
    返回:
        tuple: (frequencies, power) 频率数组和功率谱
    """
    N = len(sunspots)
    # 执行傅里叶变换
    fft_result = np.fft.fft(sunspots)
    # 计算功率谱 |c_k|^2
    power = np.abs(fft_result)**2 / N
    # 计算频率
    frequencies = np.fft.fftfreq(N, 1.0)  # 频率间隔为1个月
    # 只保留正频率部分（不包括0频率）
    positive_freq_mask = (frequencies > 0) & (frequencies <= 0.5)
    frequencies = frequencies[positive_freq_mask]
    power = power[positive_freq_mask]
    return frequencies, power

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
    """
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, power)
    plt.title('太阳黑子数据功率谱')
    plt.xlabel('频率 (1/月)')
    plt.ylabel('功率')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def find_main_period(frequencies, power):
    """
    找出功率谱中的主周期
    
    参数:
        frequencies (numpy.ndarray): 频率数组
        power (numpy.ndarray): 功率谱数组
        
    返回:
        float: 主周期（月）
    """
    # 找到最大功率对应的索引
    max_power_idx = np.argmax(power)
    # 获取对应的频率
    main_frequency = frequencies[max_power_idx]
    # 计算周期（月）
    main_period = 1.0 / main_frequency
    return main_period

def main():
    # 数据文件路径
    data = "sunspot_data.txt"
    
    # 1. 加载并可视化数据
    years, sunspots = load_sunspot_data(data)
    plot_sunspot_data(years, sunspots)
    
    # 2. 傅里叶变换分析
    frequencies, power = compute_power_spectrum(sunspots)
    plot_power_spectrum(frequencies, power)
    
    # 3. 确定主周期
    main_period = find_main_period(frequencies, power)
    print(f"\n主周期计算结果: {main_period:.2f} 月")
    print(f"约等于 {main_period/12:.2f} 年")

if __name__ == "__main__":
    main()
