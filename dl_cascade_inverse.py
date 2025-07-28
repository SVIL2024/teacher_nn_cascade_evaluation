import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from utils import set_seed,score_ditc2softmax
import numpy as np
import torch
import torch.nn as nn
import os
import numpy as np
import torch
import torch.nn as nn
# 导入必要模块
from dl_cascade_model import FlexibleHierNetwork
from dl_cascade_proc_dt import load_and_preprocess_all_sheets   
from dl_cascade_model import train_model
from dl_cascade_personal import compute_avg_scores, compute_max_scores, get_second_level_scores, plot_teacher_profile

def inverse_prediction(model,
                       processed_data,
                       target_score=0.8,
                       learning_rate=0.01,
                       num_iterations=1000):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    # 初始化参数
    best_inputs = nn.ParameterDict({
        key: nn.Parameter(
            torch.empty(1, tensor.shape[1], device=device, dtype=torch.float32).normal_(0, 0.1)  # 更小的初始化
        )
        for key, tensor in processed_data['X_train_dict'].items()
    })
    
    # 使用带衰减的优化器
    optimizer = torch.optim.Adam(best_inputs.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

    target_logit = torch.tensor([target_score], device=device, dtype=torch.float32)
    criterion = nn.MSELoss()
    loss_history = []

    for i in range(num_iterations):
        optimizer.zero_grad()
        
        logits = model(best_inputs)
        loss = criterion(logits, target_logit)
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸或消失
        torch.nn.utils.clip_grad_norm_(best_inputs.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # 学习率衰减

        loss_history.append(loss.item())

        if i % 100 == 0 and i>0:
            with torch.no_grad():
                prob = torch.sigmoid(logits).item()
                print(f"iter {i:4d} | lr={scheduler.get_last_lr()[0]:.6f} | loss={loss.item():.6f} | pred={prob:.4f}")
            
                # for param in best_inputs.parameters():
                #     noise = torch.randn_like(param) * 0.01
                #     param.add_(noise)
            # 检查梯度
            for name, param in best_inputs.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    param_norm = param.norm().item()
                    print(f"  {name} 梯度范数: {grad_norm:.6f}, 参数范数: {param_norm:.6f}")
                else:
                    print(f"  {name} 没有梯度")

    # 后续代码保持不变
        optimizer.step()
        loss_history.append(loss.item())

    model.eval()
    with torch.no_grad():
        sub_scores = {}
        for key, param in best_inputs.items():
            out = model.subnets[key](param)
            sub_scores[key] = float((out).mean().item())

    best_inputs_np = {k: v.detach().cpu().numpy() for k, v in best_inputs.items()}
    return sub_scores, best_inputs_np, loss_history

def plot_inverse_path(scores_dict, avg_scores, title="Inverse Prediction Path"):
    """
    绘制雷达图展示从目标评分反推出的二级指标得分
    """
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    avg_values = [avg_scores[cat] for cat in categories]
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    avg_values += avg_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 绘制预测得分
    ax.plot(angles, values, linewidth=1, linestyle='solid', label='Predicted Score')
    ax.fill(angles, values, color='blue', alpha=0.25)

    # 绘制平均得分
    ax.plot(angles, avg_values, linewidth=1, linestyle='dashed', label='Average Score')

    plt.xticks(angles[:-1], categories, color='grey', size=8)
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=7)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title)
    plt.show()

def print_recommendations(scores_dict, feature_contributions, level="second-level"):
    """
    打印教师提升路径建议，优化输出格式，让人更容易理解

    参数:
        scores_dict: {二级指标名: 得分} 字典
        feature_contributions: {二级指标名: [(特征名, 重要性), ...]}
        level: 提示是二级还是三级指标建议
    """
    print("\n🎯 Teacher Development Suggestion Report")
    print("=" * 60)

    for key in sorted(scores_dict.keys()):
        score = scores_dict[key]
        print(f"\n【{key}】 ({level} indicator)")
        print("→ Current score: {:.2f}".format(score))
        print("→ Recommendations：")

        contributions = feature_contributions.get(key, [])
        if not contributions:
            print("  → No recommendations")
            continue

        for item in contributions[:5]:  # 只显示前5个最重要的特征
            if isinstance(item, tuple) and len(item) == 2:
                feat, contrib = item
                # 去除二级指标前缀，只保留三级指标名
                clean_feat = feat.split('_', 1)[-1] if '_' in feat else feat
                print(f"  🔹 {clean_feat.ljust(25)} ➜ Influence the weight: {contrib:.4f}")
            else:
                print("  → Invalid feature data format. Skip this item")

    print("\n💡 Tip: The higher the value, the greater the impact. It is recommended to prioritize improving those with higher weights first")

def plot_improvement_direction(scores_dict, target_score_dict):
    """
    显示当前得分与目标得分之间的差距
    
    参数:
        scores_dict: {'Basic Background': 0.8, 'Scientific Innovation': 0.75, ...}
        target_score_dict: {'Basic Background': 0.9, 'Scientific Innovation': 0.9, ...}
    """
    categories = list(scores_dict.keys())
    current_scores = list(scores_dict.values())
    target_scores = [target_score_dict.get(cat, score) for cat, score in scores_dict.items()]
    
    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width/2, current_scores, width, label='Current Score', color='skyblue')
    rects2 = ax.bar(x + width/2, target_scores, width, label='Target Score', color='orange')

    ax.set_ylabel('Score')
    ax.set_title('Current performance vs Best Performance', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.show()

# ... existing code ...
def compute_feature_importance(model, X_train_dict, X_train_dict_names):
    """
    使用简单的梯度方法计算特征重要性
    
    参数:
        model: 训练好的模型
        X_train_dict: 训练数据字典
        X_train_dict_names: 特征名称字典
        
    返回:
        feature_importance: {二级指标名: [(特征名, 重要性), ...]}
    """
    model.eval()
    
    # 选择一个样本作为基准（这里使用每个特征的均值）
    baseline_data = {}
    for key, tensor in X_train_dict.items():
        # 计算每个特征的均值
        baseline_data[key] = torch.mean(tensor, dim=0, keepdim=True)
        baseline_data[key].requires_grad_(True)
    
    # 前向传播
    output = model(baseline_data)
    
    # 计算梯度
    model.zero_grad()
    output.backward()
    
    # 收集梯度作为重要性
    feature_importance = {}
    for key in X_train_dict_names.keys():
        if key in baseline_data and baseline_data[key].grad is not None:
            # 获取梯度的绝对值作为重要性
            importance_values = torch.abs(baseline_data[key].grad).detach().cpu().numpy()[0]
            feature_names = X_train_dict_names[key]
            
            # 组合成(特征名, 重要性)的列表
            feature_importance[key] = list(zip(feature_names, importance_values))
            
            # 按重要性排序
            feature_importance[key].sort(key=lambda x: x[1], reverse=True)
        else:
            # 如果没有梯度，使用默认值
            feature_names = X_train_dict_names[key]
            feature_importance[key] = [(name, 1.0) for name in feature_names]
    
    return feature_importance

# ... existing code ...
def plot_recommendations(scores_dict, feature_contributions, top_k=5, cols=2):
    """
    Visualize teacher development recommendations in an n*m grid format
    
    Parameters:
        scores_dict: {second-level indicator: score} dictionary
        feature_contributions: {second-level indicator: [(feature name, importance), ...]}
        top_k: Number of top features to display for each indicator
        cols: Number of charts per row
    """
    # Calculate the required number of rows and columns
    n_categories = len(scores_dict)
    rows = (n_categories + cols - 1) // cols  # Round up
    
    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 4*rows))
    
    # If there's only one subplot, convert axes to a 2D array for uniform processing
    if n_categories == 1:
        axes = np.array([[axes]])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(rows, cols)
    
    # Get sorted category list
    sorted_categories = sorted(scores_dict.items())
    
    # Create bar charts for each second-level indicator
    for idx, (category, score) in enumerate(sorted_categories):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Get feature importance for this category
        contributions = feature_contributions.get(category, [])
        
        # Take only top_k features
        top_features = contributions[:top_k] if len(contributions) >= top_k else contributions
        
        if top_features:
            # Separate feature names and importance values
            feature_names = [feat.split('_', 1)[-1] if '_' in feat else feat for feat, _ in top_features]
            importance_values = [contrib for _, contrib in top_features]
            
            # Create horizontal bar chart
            y_pos = np.arange(len(feature_names))
            bars = ax.barh(y_pos, importance_values, color='skyblue')
            
            # Set chart properties
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.set_xlabel('Feature Importance')
            ax.set_title(f'[{category}] (Current Score: {score:.2f})')
            ax.invert_yaxis()  # Most important features at the top
            
            # Add value labels on the bars
            for i, (bar, value) in enumerate(zip(bars, importance_values)):
                ax.text(bar.get_width() + max(importance_values)*0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', ha='left', va='center', fontsize=9)
        else:
            ax.text(0.5, 0.5, 'No recommendations', horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(f'[{category}] (Current Score: {score:.2f})')
        
        ax.grid(axis='x', alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_categories, rows * cols):
        row = idx // cols
        col = idx % cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.suptitle('Teacher Development Recommendations - Feature Importance Analysis', fontsize=16, y=1.02)
    plt.show()


def plot_score_comparison(scores_dict, max_scores, avg_scores):
    """
    绘制得分对比图，包括当前得分、平均得分和最高得分
    
    参数:
        scores_dict: 当前预测得分
        max_scores: 最高得分
        avg_scores: 平均得分
    """
    categories = list(scores_dict.keys())
    current_scores = [scores_dict[cat] for cat in categories]
    max_vals = [max_scores.get(cat, 0) for cat in categories]
    avg_vals = [avg_scores.get(cat, 0) for cat in categories]
    
    # 设置柱状图的位置
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制三组柱状图
    rects1 = ax.bar(x - width, current_scores, width, label='current performance', color='skyblue')
    rects2 = ax.bar(x, avg_vals, width, label='average performance', color='lightcoral')
    rects3 = ax.bar(x + width, max_vals, width, label='best performance', color='lightgreen')
    
    # 添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    # 设置图表属性
    ax.set_xlabel('Second-level indicators')
    ax.set_ylabel('Scores')
    ax.set_title('Analysis of Teacher Development Paths - Score Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.show()
# ... existing code ...

set_seed(5)
# 全局设置所有字体大小
plt.rcParams.update({
    'font.size': 14,              # 基础字体大小
    'axes.titlesize': 16,         # 标题字体大小
    'axes.labelsize': 14,         # 坐标轴标签字体大小
    'xtick.labelsize': 12,        # x轴刻度字体大小
    'ytick.labelsize': 12,        # y轴刻度字体大小
    'legend.fontsize': 12,        # 图例字体大小
    'figure.titlesize': 16        # 图形标题字体大小
})

# 1. 加载并预处理所有sheet的数据
file_path = 'teacher_info_2025_name-7-3-english.xlsx'
processed_data = load_and_preprocess_all_sheets(file_path, target_column='score')

# 2. 构建模型输入维度
input_dim_dict = {
    key: tensor.shape[1] for key, tensor in processed_data['X_train_dict'].items()
}

# 3. 创建模型
model = FlexibleHierNetwork(input_dim_dict, hidden_dims=[32, 16], output_dim=1)
if os.path.exists("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth"))
else:# 5. 训练模型（示例训练过程）
    train_model(model, processed_data['X_train_dict'], 
                processed_data['y_train'],
                processed_data['X_test_dict'], 
                processed_data['y_test'],
                epochs=500, 
                lr=1e-3)

# 6. 设置目标评分（归一化到0~1区间）
target_score = 0.5 # 表示希望达到的综合评分（归一化后）

# 7. 运行逆推路径分析
scores_dict, best_inputs, loss_history = inverse_prediction(
    model, 
    processed_data, 
    target_score=target_score, 
    learning_rate = 0.1, 
    num_iterations=2000
)
scores_dict = score_ditc2softmax(scores_dict)
# 中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 8. 获取平均得分（用于雷达图对比）
avg_scores = compute_avg_scores(model, processed_data['X_train_dict'])

# 9. 绘制雷达图
plot_inverse_path(scores_dict, avg_scores, title=f"Path to Score {target_score}")

# 10. 打印建议提升方向
feature_contributions = compute_feature_importance(
    model, 
    processed_data['X_train_dict'], 
    processed_data['X_train_dict_names'],
)
print_recommendations(scores_dict, feature_contributions)

# 11. 可视化建议提升方向
plot_recommendations(scores_dict, feature_contributions)

# 12. 绘制得分对比图
max_scores = compute_max_scores(model, processed_data['X_train_dict'])
plot_score_comparison(scores_dict, max_scores, avg_scores)

plot_improvement_direction(scores_dict, max_scores)

# 11 shap解释
# import shap
# explainer = shap.GradientExplainer(model, background_data)
