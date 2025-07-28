import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dl_cascade_model import FlexibleHierNetwork, train_model, load_and_preprocess_all_sheets
from dl_cascade_proc_dt import load_and_preprocess_all_sheets
from utils import set_seed, score_ditc2softmax
import matplotlib
# matplotlib.use('Agg')  # 设置为非交互式后端
import os
def get_second_level_scores(model, X_dict):
    """
    获取教师在每个二级指标上的得分
    
    参数:
        model: 级联神经网络模型 (CascadingModel)
        X_dict: 字典形式的输入数据 {'Basic Background': tensor, 'Scientific Innovation': tensor, ...}
        
    返回:
        scores: 二级指标得分字典 {'Basic Background': score, ...}
    """
    model.eval()
    
    with torch.no_grad():
        # 获取每个子网络的输出, 中间维度为16，所以取均值
        sub_outputs = {
            key: model.subnets[key](x).mean().item()
            for key, x in X_dict.items()
        }

    return sub_outputs

def plot_teacher_profile(scores, avg_scores, title="Teacher Self-Profile"):
    """
    绘制雷达图展示教师在各二级指标上的得分分布
    
    参数:
        scores: 当前教师的二级指标得分 {'指标名': 分数}
        avg_scores: 所有教师在这些指标上的平均得分 {'指标名': 分数}
    """
    categories = list(scores.keys())
    values = list(scores.values())
    avg_values = [avg_scores[cat] for cat in categories]
    
    N = len(categories)

    # 设置雷达图角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]  # 闭合图形
    avg_values += avg_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 绘制个人得分
    ax.plot(angles, values, linewidth=1, linestyle='solid', label='Teacher')
    ax.fill(angles, values, color='blue', alpha=0.25)

    # 绘制平均得分
    ax.plot(angles, avg_values, linewidth=1, linestyle='dashed', label='Average')

    # 添加标签
    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_rlabel_position(30)
    
    # 设置雷达图刻度
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    
    # 图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title)
    plt.show()
def compute_avg_scores(model, X_train_dict):
    """
    对训练集中的所有教师进行推理，获取每个二级指标的平均得分
    
    参数:
        model: 已训练模型
        X_train_dict: 字典形式的训练数据 {'指标名': tensor}
    
    返回:
        avg_scores: 每个二级指标的平均得分 {'指标名': 平均分数}
    """
    # all_scores = []
    
    # 获取样本数量（假设所有键下的张量行数一致）
    # num_samples = next(iter(X_train_dict.values())).shape[0]


    model.eval()
    with torch.no_grad():
        # 获取每个子网络的输出, 中间维度为16，所以取均值
        sub_outputs = {
            key: model.subnets[key](x).mean().item()
            for key, x in X_train_dict.items()
        }
   
    return sub_outputs

def compute_max_scores(model, X_train_dict):
    """
    对训练集中的所有教师进行推理，获取每个二级指标的平均得分
    
    参数:
        model: 已训练模型
        X_train_dict: 字典形式的训练数据 {'指标名': tensor}
    
    返回:
        avg_scores: 每个二级指标的平均得分 {'指标名': 平均分数}
    """


    model.eval()
    with torch.no_grad():
        # 获取每个子网络的输出, 中间维度为16，所以取均值
        sub_outputs = {
            key: model.subnets[key](x).mean(dim=1).max().item()
            for key, x in X_train_dict.items()
        }

    # all_scores = []
    
    # # 获取样本数量（假设所有键下的张量行数一致）
    # num_samples = next(iter(X_train_dict.values())).shape[0]
    
    # for i in range(num_samples):
    #     X_teacher_dict = {key: x[i:i+1] for key, x in X_train_dict.items()}
    #     scores = get_second_level_scores(model, X_teacher_dict)
    #     all_scores.append(scores)
    
    # # 将列表转换为 DataFrame 或直接求均值
    # keys = all_scores[0].keys()
    # avg_scores = {
    #     key: np.max([s[key] for s in all_scores]) 
    #     for key in keys
    # }
    
    return (sub_outputs)
if __name__ == "__main__":
    set_seed(5) 
    # 数据加载与预处理
    file_path = "teacher_info_2025_name-7-3-english.xlsx"
    processed_data = load_and_preprocess_all_sheets(file_path, target_column='score')

    # 构建输入维度字典
    input_dim_dict = {k: v.shape[1] for k, v in processed_data['X_train_dict'].items()}

    # 构建模型
    model = FlexibleHierNetwork(input_dim_dict, hidden_dims=[32, 16], output_dim=1)

    # 训练模型
    # 如果模型存在，则加载模型
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth"))
    else:
        train_model(
            model,
            processed_data['X_train_dict'],
            processed_data['y_train'],
            processed_data['X_test_dict'],
            processed_data['y_test'],
            epochs=500,
            lr=1e-2
        )

    # 测试模型输出
    model.eval()
    with torch.no_grad():
        y_pred = model(processed_data['X_test_dict'])
        print("预测输出示例:", y_pred[:5].squeeze().cpu().numpy())

    # 直接使用process_data返回的特征名字典
    feature_names_dict = processed_data['X_test_dict_names']

    # 假设 teacher_index 是你选择的教师索引
    teacher_index = 21

    # 构建该教师的输入字典
    X_teacher_dict = {key: processed_data['X_train_dict'][key][teacher_index] for key in processed_data['X_test_dict']}

    # 获取该教师的二级指标得分
    scores = get_second_level_scores(model, X_teacher_dict)

    # 获取平均得分
    avg_scores = compute_avg_scores(model, processed_data['X_train_dict'])

    # 获取最高得分
    max_scores = compute_max_scores(model, processed_data['X_train_dict'])
    print(f'current teacher scores: {scores}, average scores: {avg_scores}')
    # 绘图
    # plt.title(f"Teacher {teacher_index} Profile")
    plot_teacher_profile(scores, avg_scores, title=f"Teacher {teacher_index} self-Profile")
    plt.draw()
    plt.savefig("teacher_profile.png",dpi=300,bbox_inches='tight')