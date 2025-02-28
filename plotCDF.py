from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load the MovieLens 100k dataset using RecBole
config_dict = {
    'model': 'BPR',  # Model name is required by RecBole, but we don't need to train a model for this task
    'dataset': 'ml-1m',  # Adjust the path to where your ml-100k dataset is located
}
config = Config(model='BPR', config_dict=config_dict)
dataset = create_dataset(config)

# Step 2: Prepare the training data
train_data, valid_data, test_data = data_preparation(config, dataset)
# 计算不同组合的交互数量 #train_data.interaction_matrix.toarray()
interactions = dataset.inter_matrix(form="coo").astype(np.float32).todense()
# 确保这是一个稀疏矩阵
# Step 3: Count item frequencies in the training data
item_counter = dataset.item_counter

item_ids, item_freqs = zip(*item_counter.items())

# Sort items by frequency
# 转换为int类型的数组
int_array = np.array([x[0] if isinstance(x, tuple) else x for x in item_freqs])
sorted_indices = np.argsort(-int_array)
sorted_item_freqs = np.array(item_freqs)[sorted_indices]
sorted_item_ids = np.array(item_ids)[sorted_indices]

# Compute the cumulative distribution
cdf = np.cumsum(sorted_item_freqs) / np.sum(sorted_item_freqs)

# Step 4: Plot the CDF
# 使用 Matplotlib 自带的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
plt.figure(figsize=(10, 8))
plt.plot(np.arange(len(cdf)), cdf, label='CDF')
plt.plot([0, len(cdf)], [0, 1], 'k--', label='Random')
#刻度值字体大小设置（x轴和y轴同时设置）
plt.tick_params(labelsize=24)
# plt.plot(sorted_item_ids, cdf)
plt.xlabel('Item ID',fontsize=24)
plt.ylabel('CDF',fontsize=24)
# plt.title('Douban-Book')
plt.grid(True)
plt.savefig('./ML-1M.png', dpi=600)
plt.show()

