#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
太阳黑子周期性分析 - 修正版代码
"""

import numpy as np
import matplotlib.pyplot as plt

def load_sunspot_data(url):
    """
    从本地文件读取太阳黑子数据（修正列数不一致问题）
    
    参数:
        url (str): 本地文件路径
        
    返回:
        tuple: (years, sunspots) 年份和太阳黑子数
    """
    # 使用usecols指定读取第2列（年份，索引1）和第4列（太阳黑子数，索引3）
    data = np.loadtxt(url, usecols=(1, 3))
    years = data[:, 0]
    sunspots = data[:, 1]
    
    # 处理缺失值（太阳黑子数为-1或0可能表示缺失，根据数据规则过滤）
    mask = sunspots > 0  # 保留有效正值数据
    years = years[mask]
    sunspots = sunspots[mask]
    
    return years, sunspots

def plot_sunspot_data(years, sunspots):
    """
    绘制太阳黑子数据随时间变化图
    """
    plt.figure(figsize=(12, 6))
    plt.plot(years, sunspots, linestyle='-', linewidth=0.8)
    plt.title('太阳黑子数随时间变化（1749-2025）', fontsize=14)
    plt.xlabel('年份', fontsize=12)
    plt.ylabel('太阳黑子数', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    # 移除用户输入估计周期的交互部分（测试时不需要手动输入）
    # estimated_period = float(input("请根据图像估计周期（月）："))
    # print(f"目视估计周期：{estimated_period} 月")

def compute_power_spectrum(sunspots):
    """
    计算太阳黑子数据的功率谱
    """
    N = len(sunspots)
    fft_result = np.fft.fft(sunspots)
    power = np.abs(fft_result) ** 2 / N  # 归一化功率谱
    frequencies = np.fft.fftfreq(N, d=1)  # 频率间隔为1个月（d=1）
    
    # 提取正频率部分（排除零频率和冗余负频率）
    positive_mask = frequencies > 0
    frequencies = frequencies[positive_mask]
    power = power[positive_mask]
    
    return frequencies, power

def plot_power_spectrum(frequencies, power):
    """
    绘制功率谱图并标记主周期
    """
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, power, label='功率谱密度')
    plt.title('太阳黑子数据功率谱分析', fontsize=14)
    plt.xlabel('频率（1/月）', fontsize=12)
    plt.ylabel('功率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 找到最大功率对应的频率和周期
    max_idx = np.argmax(power)
    main_freq = frequencies[max_idx]
    main_period = 1 / main_freq
    plt.plot(main_freq, power[max_idx], 'ro', label=f'主周期: {main_period:.2f} 月')
    plt.legend()
    plt.tight_layout()
    plt.show()

def find_main_period(frequencies, power):
    """
    提取主周期
    """
    max_idx = np.argmax(power)
    main_freq = frequencies[max_idx]
    return 1 / main_freq

def main():
    data_file = "sunspot_data.txt"
    
    # 1. 加载数据
    years, sunspots = load_sunspot_data(data_file)
    
    # 2. 绘制时间序列图
    plot_sunspot_data(years, sunspots)
    
    # 3. 计算并绘制功率谱
    frequencies, power = compute_power_spectrum(sunspots)
    plot_power_spectrum(frequencies, power)
    
    # 4. 确定主周期
    main_period = find_main_period(frequencies, power)
    print(f"主周期计算结果: {main_period:.2f} 个月 ≈ {main_period/12:.2f} 年")

if __name__ == "__main__":
    main()
