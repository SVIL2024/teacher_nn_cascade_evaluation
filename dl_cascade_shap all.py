import shap
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Optional
import os
import time
from utils import set_seed
set_seed(5)  # 设置随机种子以确保结果可复现
# 处理中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def explain_with_shap(model,
                      X_dict: Dict[str, torch.Tensor],
                      feature_names_dict: Dict[str, List[str]],
                      nsamples=100):
    """
    对 FlexibleHierNetwork 模型进行全局 SHAP 解释（按子网络）
    
    参数:
        model: 已训练的模型
        X_dict: 测试集输入（dict，每个 key 是二级指标名，value 是 tensor）
        feature_names_dict: dict，每个 key 是二级指标名，value 是特征名列表
        nsamples: 背景数据采样数量
    
    返回:
        shap_values_dict: 每个子网络的 SHAP 值
    """
    # 转为 numpy
    X_np_dict = {k: v[:nsamples].cpu().numpy() for k, v in X_dict.items()}
    shap_values_dict = {}

    def subnet_forward(x, key):
        x_torch = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return model.subnets[key](x_torch).cpu().numpy()

    for key in X_np_dict:
        print(f"\n🔍 正在对子网络 '{key}' 进行 SHAP 解释...")

        # ✅ 使用 PermutationExplainer（PyTorch 兼容）
        explainer = shap.PermutationExplainer(lambda x: subnet_forward(x, key), X_np_dict[key])

        try:
            shap_values = explainer(X_np_dict[key])
            shap_values_dict[key] = shap_values.values[:, :, 0]  # 取第一个输出维度

            # 绘图
            print(f"SHAP summary plot for {key}:")
            plt.title(f"SHAP Summary Plot for {key}")
            shap.summary_plot(shap_values.values[..., 0], X_np_dict[key], 
                              feature_names=feature_names_dict[key],
                              max_display=10)
            # plt.tight_layout()
            plt.pause(0.1)

            plt.savefig(os.path.join('dl_cascade/data', f"shap_summary_{key}.png"),
                        dpi=300, bbox_inches='tight')
            plt.clf()  # 清除当前图形以便下一个子网络的绘图
            # plt.show()
        except Exception as e:
            print(f"[ERROR] 子网络 '{key}' SHAP 解释失败:", e)

    return shap_values_dict


def explain_single_instance(model,
                            processed_data,
                            teacher_index=0,
                            nsamples=100,
                            show_summary=True,
                            show_waterfall=True,
                            show_force=True,
                            save_path: Optional[str] = None):
    """
    使用 SHAP 对单个教师的预测结果进行解释（仅使用 PermutationExplainer）

    参数:
        model: 已训练的模型
        processed_data: 包含 X_test_dict 和 X_test_dict_names 的预处理数据
        teacher_index: 教师索引
        nsamples: 采样数量（用于 SHAP 背景数据）
        show_summary: 是否显示 summary plot
        show_waterfall: 是否显示 waterfall plot
        show_force: 是否显示 force plot
        save_path: 图像保存路径（如果提供则自动保存）

    返回:
        shap_df: 包含特征名、SHAP值和原始输入值的 DataFrame
    """
    # 设置中文支持
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 获取该教师的输入数据（dict of tensors）
    X_teacher_dict = {
        key: processed_data['X_test_dict'][key][teacher_index:teacher_index+1]
        for key in processed_data['X_test_dict']
    }

    # 合并为 flat 输入张量（用于 SHAP）
    all_inputs = []
    feature_names = []

    for key in X_teacher_dict:
        input_tensor = X_teacher_dict[key].cpu().numpy().squeeze(0)
        all_inputs.extend(input_tensor.tolist())

        # 获取当前二级指标下的特征名
        names = processed_data['X_test_dict_names'][key]
        current_features = []

        if isinstance(names, list):
            current_features = [name.split('_', 1)[-1] if '_' in name else name for name in names]
        else:
            current_feature = str(names).split('_', 1)[-1]
            current_features.append(current_feature)

        assert len(current_features) == X_teacher_dict[key].shape[1], \
            f"特征名与维度不匹配：{key} ({len(current_features)} != {X_teacher_dict[key].shape[1]})"

        feature_names.extend(current_features)

    full_teacher_input = np.array(all_inputs).reshape(1, -1)
    print("合并后输入形状:", full_teacher_input.shape)
    print("特征数量:", len(feature_names))

    # 构建背景数据
    background_list = []
    for key in processed_data['X_test_dict']:
        background_list.append(processed_data['X_test_dict'][key][:nsamples].cpu().numpy())
    background_data = np.hstack(background_list)
    print("背景数据形状:", background_data.shape)

    # 定义前向函数
    def model_forward(x):
        x_torch = torch.tensor(x, dtype=torch.float32)
        inputs = {}
        start_idx = 0
        for key in X_teacher_dict:
            dim = X_teacher_dict[key].shape[1]
            inputs[key] = x_torch[:, start_idx:start_idx + dim]
            start_idx += dim
        with torch.no_grad():
            return model(inputs).cpu().numpy()

    # ✅ 创建 PermutationExplainer（仅使用 PyTorch）
    # permutionExplainer更适合用于全局特征重要性评估，快速了解哪些特征对模型的整体性能影响最大
    #  explainer 是一个解释器对象，它的主要功能是计算 SHAP 值。SHAP 值是通过特定的算法（如树模型的 
    # Tree SHAP、通用模型的 Kernel SHAP 等）计算出来的，这些算法需要对模型和数据进行特定的处理。
    explainer = shap.PermutationExplainer(model_forward, background_data)

    # 计算 SHAP 值
    shap_values = explainer(full_teacher_input)
    expected_value = model_forward(full_teacher_input).item() - shap_values.base_values[0]

    print(f"Expected Value (base_value): {expected_value}")

    # 构建 Explanation 对象,对象提供了一种标准化的方式来存储和操作 SHAP 值，使得这些值可以被不同的可视化函数
    #  是 SHAP 库中用于存储解释结果的对象,更适合用于局部解释，详细分析单个样本的预测结果，了解每个特征对预测的具体贡献
    explanation = shap.Explanation(
        values=shap_values.values[0],
        base_values=np.array([expected_value]),
        data=full_teacher_input[0],
        feature_names=feature_names
    )

    # 显示图表
    if show_summary:
        print("\n📊 Summary Plot:")
        # only one teacher data, so for one teacher, but 
        # summary_plot is also used for global feature importance
        # shap.summary_plot(explanation.values, explanation.data, feature_names=explanation.feature_names)
        shap.summary_plot(shap_values, full_teacher_input, feature_names=explanation.feature_names,max_display=10)
        if save_path:
            plt.savefig(f"{save_path}_summary.png")
        plt.show()

    if show_waterfall:
        print("\n📉 Waterfall Plot:")
        # only show one teacher 
        '''
        用于展示单个样本的 SHAP 值（SHapley Additive exPlanations）及其对模型预测的贡献。
        通过瀑布图，你可以直观地看到每个特征对模型预测的具体影响，以及这些影响是如何累加起来的。
       1 解释单个预测：
        瀑布图可以帮助你理解模型对单个样本的预测是如何由各个特征贡献的。你可以看到每个特征对最终预测值的正向或负向影响。
       2 特征重要性：
        瀑布图可以直观地展示哪些特征对模型预测的影响最大。特征按其对预测值的贡献大小排序，你可以清楚地看到哪些特征是关键驱动因素。
       3 模型调试：
        通过观察特征的贡献，你可以发现模型可能存在的问题，例如某些特征的贡献是否合理，或者是否存在异常值对预测的影响。
        '''
        shap_values.feature_names = explanation.feature_names
        shap.plots.waterfall(shap_values[0],
                             max_display=10)
        if save_path:
            plt.savefig(f"{save_path}_waterfall.png")
        plt.show()

    if show_force:
        print("\n📈 Force Plot:")
        '''
        用于展示单个样本的 SHAP 值及其对模型预测的贡献。力图通过颜色和条形的长度来表示每个
        特征的正向或负向影响，非常直观地展示了特征对模型预测的作用。
        1 解释单个预测：力图可以帮助你理解模型对单个样本的预测是如何由各个特征贡献的。
        你可以看到每个特征对最终预测值的正向或负向影响。
        2 特征影响：力图可以直观地展示哪些特征对模型预测的影响最大。特征按其对预测值的贡献大小排序，
        你可以清楚地看到哪些特征是关键驱动因素。
        3 交互式可视化：力图支持交互式功能，你可以通过鼠标悬停或点击来查看详细信息，
        这使得解释过程更加动态和直观。
        '''
        shap.force_plot(base_value = explanation.base_values[0], #模型的基础值，通常为 explainer.expected_value，表示模型预测的平均值或期望值。
                        shap_values = explanation.values, #单个样本的 SHAP 值，是一个数组，表示每个特征对该样本预测的贡献。
                        features= explanation.data,#单个样本的特征值，是一个数组或列表。
                        feature_names=explanation.feature_names, #特征名称列表，用于在图表中显示特征名称
                        link='identity',# 链接函数，用于将 SHAP 值转换为输出值。默认为 'identity'，表示直接使用 SHAP 值。
                        plot_cmap='RdBu', #颜色映射，用于表示特征的正向和负向影响。默认为 'RdBu'
                        out_names= 'force plot',# 输出名称，用于显示在图表的标题中
                        matplotlib=True)
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_force.png")
        plt.show()

    # 构建结果 DataFrame
    shap_df = pd.DataFrame({
        'features': explanation.feature_names,
        'shap_values': explanation.values,
        'input_values': explanation.data
    })
    shap_df = shap_df.sort_values(by='shap_values', key=lambda x: abs(x), ascending=False)

    return shap_df

if __name__ == "__main__":
    from dl_cascade_proc_dt import load_and_preprocess_all_sheets
    from dl_cascade_model import FlexibleHierNetwork, train_model

    # 加载数据
    file_path = r"E:\program\python\paper\teacher_analysis\dl_cascade\teacher_info_2025_name-7-3-english.xlsx"
    processed_data = load_and_preprocess_all_sheets(file_path, target_column='score')

    # 构建模型
    input_dim_dict = {k: v.shape[1] for k, v in processed_data['X_train_dict'].items()}
    model = FlexibleHierNetwork(input_dim_dict, hidden_dims=[32, 16], output_dim=1)

    # 训练模型
    train_model(
        model,
        processed_data['X_train_dict'],
        processed_data['y_train'],
        processed_data['X_test_dict'],
        processed_data['y_test'],
        epochs=500,
        lr=1e-3
    )

    # 测试输出
    model.eval()
    with torch.no_grad():
        y_pred = model(processed_data['X_test_dict'])
        print("预测输出示例:", y_pred[:5].squeeze().cpu().numpy())

    # 根据训练数据，对模型输出对应的特征进行解释
    explain_with_shap(model, processed_data['X_test_dict'], processed_data['X_test_dict_names'])

    # 针对某一个教师的数据，解释和说明
    # SHAP 解释
    # teacher_index = 0
    # shap_df = explain_single_instance(
    #     model,
    #     processed_data,
    #     teacher_index=teacher_index,
    #     nsamples=100,
    #     # explainer_type="gradient",
    #     show_summary=True,
    #     show_waterfall=True,
    #     show_force=True,
    #     save_path=f"shap_teacher_{teacher_index}"
    # )

    # print("\n📊 SHAP 解释结果：")
    # print(shap_df.head(20))