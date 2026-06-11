import torch
import torch.nn as nn
import time
import numpy as np
import os
import sys

# 1. 解决路径解析问题 (针对你的目录结构)
current_script_path = os.path.abspath(__file__)  # benchmark_networks 目录
benchmark_dir = os.path.dirname(current_script_path)
code_dir = os.path.dirname(benchmark_dir)  # code 目录

if code_dir not in sys.path:
    sys.path.append(code_dir)

# 2. 修正导入语句
from Network_structure import fcNN, RNN_lstm, Simple_CNN, Complex_CNN

try:
    from Novel_CNN.Novel_CNN import Novel_CNN
    from Mida_Net.Mida_Net import MS_MidaNet_ZeroPhase

    print("✅ 所有模型模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败，请检查文件夹名。错误信息: {e}")


# 3. 修正变量名隐藏问题 (将内部参数名改为 _device)
def measure_latency(model_instance, input_shape=(1, 1, 512), _device='cuda', repetitions=1000):
    model_instance.to(_device)
    model_instance.eval()

    dummy_input = torch.randn(input_shape).to(_device)

    # 预热
    with torch.no_grad():
        for _ in range(100):
            _ = model_instance(dummy_input)

    # 计时
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model_instance(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    return np.mean(timings), np.std(timings)


if __name__ == "__main__":
    # 使用字符串或对象均可，为了消除 PyCharm 警告，我们统一定义
    RUN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    datanum = 512

    # 实例化所有模型
    models_to_test = {
        "fcNN": fcNN(datanum),
        "RNN_lstm": RNN_lstm(datanum),
        "Simple_CNN": Simple_CNN(datanum),
        "Complex_CNN": Complex_CNN(datanum),
        "Novel_CNN": Novel_CNN(datanum),
        "MP_Net (Ours)": MS_MidaNet_ZeroPhase(base_c=24)
    }

    latency_results = {}

    print(f"--- 🚀 3090 推理延迟测试 ---")

    for name, m_obj in models_to_test.items():
        avg, std = measure_latency(m_obj, _device=RUN_DEVICE)
        latency_results[name] = avg
        print(f"[{name:<18}] 平均延迟: {avg:.4f} ms")

    # 打印最终对比表
    print("\n" + "=" * 40)
    print(f"{'Model':<20} | {'Latency (ms)':<15}")
    print("-" * 40)
    for name, lat in latency_results.items():
        print(f"{name:<20} | {lat:<15.4f}")
    print("=" * 40)