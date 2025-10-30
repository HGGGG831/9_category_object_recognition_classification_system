import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# 原始数据
light_power = np.array([2.62, 4.286, 5.9524, 7.43, 9.524, 10.476, 13.12, 14.2857, 15.5, 16.69, 17.85, 19.12])
light_current = np.array([476.3, 536.076, 551.65, 566.27, 590.44, 591.8, 622.56, 630, 638.79, 646.112, 652.794, 659.5])
pressure_kpa = np.array(
    [0.98, 1.96, 3.926, 5.877, 8.0164, 9.81764, 11.77, 13.72, 15.6712, 17.7, 19.6, 21.6, 23.6, 25.5, 27.46, 29.4, 31.4,
     33.3, 35.3, 37.287, 39.2])
pressure_current = np.array(
    [13.324, 17.593, 23.243, 27.323, 30.95, 33.773, 36.1, 39.38, 40.486, 42.51, 44.285, 46.0455, 47.1, 49.2157, 50.7,
     52.31, 53.5, 54.78, 56.065, 57.311, 58.47])

# 类别定义
class_rules = {
    "薄塑料袋": {"light_range": (0.7, 1.0), "pressure_range": (0.0, 0.3)},
    "透明塑料瓶": {"light_range": (0.7, 1.0), "pressure_range": (0.3, 0.7)},
    "玻璃杯": {"light_range": (0.7, 1.0), "pressure_range": (0.7, 1.0)},
    "亚克力盒": {"light_range": (0.3, 0.7), "pressure_range": (0.0, 0.3)},
    "磨砂玻璃杯": {"light_range": (0.3, 0.7), "pressure_range": (0.3, 0.7)},
    "硬塑料盒": {"light_range": (0.3, 0.7), "pressure_range": (0.7, 1.0)},
    "橡皮擦": {"light_range": (0.0, 0.3), "pressure_range": (0.0, 0.3)},
    "木块": {"light_range": (0.0, 0.3), "pressure_range": (0.3, 0.7)},
    "金属小物件": {"light_range": (0.0, 0.3), "pressure_range": (0.7, 1.0)}
}

# 标准化器
scaler_light = MinMaxScaler().fit(light_current.reshape(-1, 1))
scaler_pressure = MinMaxScaler().fit(pressure_current.reshape(-1, 1))

# 反标准化 + 插值函数（加噪声+clip）
def generate_uniform_data_per_class(class_info, samples_per_class=100):
    light_min, light_max = class_info["light_range"]
    pressure_min, pressure_max = class_info["pressure_range"]

    # 为避免接近边界，略微收紧区间
    margin = 0.02
    norm_light = np.random.uniform(light_min + margin, light_max - margin, samples_per_class)
    norm_pressure = np.random.uniform(pressure_min + margin, pressure_max - margin, samples_per_class)

    # 反标准化回电流值
    light_curr = scaler_light.inverse_transform(norm_light.reshape(-1, 1)).flatten()
    pressure_curr = scaler_pressure.inverse_transform(norm_pressure.reshape(-1, 1)).flatten()

    # 添加噪声并clip，防止越界插值
    light_curr = np.clip(light_curr * (1 + np.random.normal(0, 0.01, samples_per_class)),
                         light_current.min(), light_current.max())
    pressure_curr = np.clip(pressure_curr * (1 + np.random.normal(0, 0.015, samples_per_class)),
                            pressure_current.min(), pressure_current.max())

    # 插值计算物理量
    light_pwr = np.interp(light_curr, light_current, light_power)
    pressure_k = np.interp(pressure_curr, pressure_current, pressure_kpa)

    # 物理量也加轻微噪声（模拟非理想插值）
    light_pwr *= (1 + np.random.normal(0, 0.02, samples_per_class))
    pressure_k *= (1 + np.random.normal(0, 0.03, samples_per_class))

    return light_pwr, light_curr, pressure_k, pressure_curr

# 合成数据
all_data = []
for class_name, rules in class_rules.items():
    l_pwr, l_curr, p_kpa, p_curr = generate_uniform_data_per_class(rules, 80)
    df = pd.DataFrame({
        "功率密度_mW/cm2": l_pwr,
        "光照电流_nA": l_curr,
        "压强_KPa": p_kpa,
        "压力电流_nA": p_curr,
        "类别": class_name
    })
    all_data.append(df)

# 合并与打乱
augmented_data = pd.concat(all_data, ignore_index=True)
augmented_data = augmented_data.sample(frac=1, random_state=42).reset_index(drop=True)

# 数据可视化
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
sns.scatterplot(x='功率密度_mW/cm2', y='光照电流_nA', hue='类别', data=augmented_data, palette='tab10', alpha=0.7)
plt.title("光照特性分布")
plt.grid(True)

plt.subplot(1, 3, 2)
sns.scatterplot(x='压强_KPa', y='压力电流_nA', hue='类别', data=augmented_data, palette='tab10', alpha=0.7)
plt.title("压力特性分布")
plt.grid(True)

plt.subplot(1, 3, 3)
augmented_data['类别'].value_counts().plot(kind='bar', color='skyblue')
plt.title("类别分布")
plt.xticks(rotation=45)
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('uniform_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 数据保存
augmented_data.to_csv("uniform_9class_dataset.csv", index=False)

# 打印信息
print("✅ 均匀分布的九类数据已保存为 uniform_9class_dataset.csv")
print(f"✅ 总数据量: {len(augmented_data)} 条")
print("\n📊 类别分布统计:")
print(augmented_data['类别'].value_counts())

# 打印物理量范围检查
print("\n📏 数据范围检查：")
print(f"光照功率密度: {augmented_data['功率密度_mW/cm2'].min():.2f} ~ {augmented_data['功率密度_mW/cm2'].max():.2f}")
print(f"压强 KPa: {augmented_data['压强_KPa'].min():.2f} ~ {augmented_data['压强_KPa'].max():.2f}")
