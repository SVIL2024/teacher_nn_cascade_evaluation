import torch
import torch.nn as nn
from dl_cascade_proc_dt import load_and_preprocess_all_sheets
from utils import set_seed

# set_seed(15)
class FlexibleHierNetwork(nn.Module):
    """
    支持任意层级结构的灵活神经网络
    输入: dict，每个key为二级指标名，value为对应特征张量
    """
    def __init__(self, input_dim_dict, hidden_dims=[32, 16], output_dim=1):
        super().__init__()
        self.input_dim_dict = input_dim_dict
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # 为每个二级指标构建独立的子网络
        self.subnets = nn.ModuleDict()
        for key, in_dim in input_dim_dict.items():
            self.subnets[key] = nn.Sequential(
                nn.Linear(in_dim, hidden_dims[0]),
                nn.BatchNorm1d(hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.BatchNorm1d(hidden_dims[1]),
                nn.ReLU(),
                # nn.Sigmoid()
            )
        # 汇总所有二级指标输出，最终输出
        self.final_layer = nn.Sequential(
            nn.Linear(len(input_dim_dict) * hidden_dims[1], output_dim)
            # ,nn.Sigmoid()  # Sigmoid 激活函数用于二分类
        )

    def forward(self, x_dict):
        # x_dict: {二级指标名: tensor(batch, feat_dim)}
        outs = []
        for key in self.input_dim_dict:
            outs.append(self.subnets[key](x_dict[key]))
        concat = torch.cat(outs, dim=1)
        return self.final_layer(concat)
class FlexibleHierConvNet(nn.Module):
    """
    用 1×1 卷积替换全连接的层级网络
    输入: dict{二级指标名: (B, C_in)}
    输出: (B, output_dim)
    """
    def __init__(self, input_dim_dict, hidden_chs=[32, 16], output_dim=1):
        super().__init__()
        self.input_dim_dict = input_dim_dict
        self.hidden_chs = hidden_chs
        self.output_dim = output_dim

        # 为每个二级指标构建独立的 1×1 卷积子网络
        self.subnets = nn.ModuleDict()
        for key, in_ch in input_dim_dict.items():
            self.subnets[key] = nn.Sequential(
                nn.Conv1d(in_ch, hidden_chs[0], kernel_size=1),  # 1×1 conv
                nn.ReLU(),
                nn.Conv1d(hidden_chs[0], hidden_chs[1], kernel_size=1),
                nn.ReLU(),
                # nn.Sigmoid()
            )

        # 汇总
        self.final_conv = nn.Conv1d(len(input_dim_dict) * hidden_chs[1],
                                    output_dim,
                                    kernel_size=1)

    def forward(self, x_dict):
        # x_dict: {name: (B, C)}
        outs = []
        for key in self.input_dim_dict:
            # 1×1 卷积需要 (B, C, L)，这里 L=1
            feat = x_dict[key].unsqueeze(-1)           # (B, C, 1)
            out = self.subnets[key](feat)              # (B, hidden_chs[-1], 1)    
            outs.append(out)

        concat = torch.cat(outs, dim=1)                # (B, K*hidden_chs[-1], 1)
        out = self.final_conv(concat).squeeze(-1)      # (B, output_dim)
        return out
def train_model(model, X_train_dict, y_train, X_test_dict, y_test, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # use the bce loss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    max_acc = 0.0
    max_acc_epoch = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_dict)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_test_dict)
                val_loss = criterion(val_pred, y_test)
                # 计算准确率
                val_prob = torch.sigmoid(val_pred)
                # val_prob = val_pred
                val_pred_label = (val_prob > 0.5).float()
                acc = (val_pred_label == y_test).float().mean().item()
                if acc > max_acc:
                    max_acc = acc
                    max_acc_epoch = epoch + 1
                    torch.save(model.state_dict(), 'best_model.pth')
            print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {acc}, Val MaxAcc: {max_acc:.4f}, Val MaxAccEpoch: {max_acc_epoch}")
            


if __name__ == "__main__":
    set_seed(5)
    # 数据加载与预处理
    file_path = "teacher_info_2025_name-7-3-english.xlsx"
    processed_data = load_and_preprocess_all_sheets(file_path, target_column='score')

    # 构建输入维度字典
    input_dim_dict = {k: v.shape[1] for k, v in processed_data['X_train_dict'].items()}

    # 构建模型
    model = FlexibleHierNetwork(input_dim_dict, hidden_dims=[32, 16], output_dim=1)
    model_conv = FlexibleHierConvNet(input_dim_dict, hidden_chs=[32, 16], output_dim=1)

    print(f'FC模型参数:{sum(p.numel() for p in model.parameters())}')
    # 训练模型
    print(f"使用全连接模型进行训练".center(100, '-'))
    train_model(
        model,
        processed_data['X_train_dict'],
        processed_data['y_train'],
        processed_data['X_test_dict'],
        processed_data['y_test'],
        epochs=500,
        lr=0.01
    )
    print(f"使用卷积模型进行训练".center(100, '-'))
    print("Conv模型参数:", sum(p.numel() for p in model_conv.parameters()))
    train_model(
        model_conv,
        processed_data['X_train_dict'],
        processed_data['y_train'],
        processed_data['X_test_dict'],
        processed_data['y_test'],
        epochs=100,
        lr=0.001
    )

    # 测试模型输出
    model.eval()
    with torch.no_grad():
        y_pred = model(processed_data['X_test_dict'])
        print("预测输出示例:", y_pred[:5].squeeze().cpu().numpy())