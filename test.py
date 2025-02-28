import torch
from recbole.quick_start import run_recbole
from collections import Counter
import numpy as np
import pandas as pd

import RecModel
from RecModel import CustomLightGCN, NewTrainer, CustomLightGCN2
from recbole.config import Config
from recbole.utils import init_logger, init_seed
from recbole.data import create_dataset, data_preparation
from recbole.trainer import Trainer
from logging import getLogger
from recbole.model.general_recommender import LightGCN

config_dict = {
    # dataset config
    'field_separator': "\t",  #指定数据集field的分隔符
    'seq_separator': " " ,  #指定数据集中token_seq或者float_seq域里的分隔符
    'USER_ID_FIELD': 'user_id' ,#指定用户id域
    'ITEM_ID_FIELD': 'item_id', #指定物品id域
    'RATING_FIELD': 'rating',  # 指定打分rating域
    'TIME_FIELD': 'timestamp',  # 指定时间域
    'load_col': {
        'inter': ['user_id', 'item_id', 'rating','timestamp']
    },
    'val_interval': {
        'rating': [4,5]  # Filter interactions with rating equal to 5
    },
    'NEG_PREFIX': 'neg_',   #指定负采样前缀
    'leave_one_out': False,
    # training settings
    'embedding_size': 32,
    'n_layers': 3,
    'reg_weight': 1e-5,
    'epochs': 100,  #训练的最大轮数
    'learner': 'adam', #使用的pytorch内置优化器
    'learning_rate': 0.002, #学习率
    'eval_step': 10000, #每次训练后做evalaution的次数
    'stopping_step': 1, #控制训练收敛的步骤数，在该步骤数内若选取的评测标准没有什么变化，就可以提前停止了
    'group_by_user': True, #是否将一个user的记录划到一个组里，当eval_setting使用RO_RS的时候该项必须是True
    'split_ratio': {'RS': [0.7,0.1,0.2]}, #切分比例 ,"Precision","TailPercentage", "MRR"
    'metrics': ["TailRecall","HeadRecall","TailNDCG","HeadNDCG","TailPercentage","ItemCoverage"], #评测标准
    'topk': [20], #评测标准使用topk，设置成10评测标准就是["Recall@10", "MRR@10", "NDCG@10", "Hit@10", "Precision@10"]
    'valid_metric': 'HeadRecall@20', #选取哪个评测标准作为作为提前停止训练的标准
    "gpu": 0,
    'eval_batch_size': 1024,
    'train_batch_size': 1024,
    'tail_ratio': 0.8,
    'gamma': 1,
    't': 1.8,
    'train_strategy': 'GODE',
    # evalution settings
    'eval_args': {
        'split': {
            'RS': [0.7, 0.1, 0.2]
        },
        'group_by': 'user',
        'order': 'RO'
        #        'mode': {'valid': 'uni100', 'test': 'uni100'}
    }
}



# #kgrec-music modcloth ml-100k ml-1m:0.1333
#CustomLightGCN2


config = Config(model='LightGCN', dataset='douban_book', config_dict=config_dict)
init_seed(config['seed'], config['reproducibility'])
# logger initialization
init_logger(config)
logger = getLogger()

# logger.info(config)
# dataset filtering
dataset = create_dataset(config)
logger.info(dataset)

# Calculate the threshold for head items (top 20%)
total_items = dataset.item_counter
item_counts = list(total_items.items())


total_users = dataset.user_counter
user_counts = list(total_users.items())

# Sort items based on counts (descending order)
sorted_items = sorted(item_counts, key=lambda x: x[1], reverse=True)
sorted_users = sorted(user_counts, key=lambda x: x[1], reverse=True)

# Calculate the threshold for head items (top 20%)
total_items = len(sorted_items)
top_20_percent_threshold = int(total_items * 0.2)

total_users = len(sorted_users)
top_20_percent_threshold_u = int(total_users * 0.8)


# Determine head items and tail items
head_items = [item_id for item_id, count in sorted_items[:top_20_percent_threshold]]
tail_items = [item_id for item_id, count in sorted_items[top_20_percent_threshold:]]

head_users = [item_id for item_id, count in sorted_users[:top_20_percent_threshold_u]]
tail_users = [item_id for item_id, count in sorted_users[top_20_percent_threshold_u:]]


# 更新或添加多个键值对
config_dict.update({
    "tail_items": tail_items,
    "head_items": head_items, # 更新已存在的键
    "tail_users": tail_users,
    "head_users": head_users  # 更新已存在的键
})
#CustomLightGCN2 'LightGCN'
config = Config(model='LightGCN', dataset='douban_book', config_dict=config_dict)
init_logger(config)
logger = getLogger()

logger.info(config)





# run_recbole(model='LightGCN', dataset='douban_book', config_dict=config_dict)
# dataset splitting
train_data, valid_data, test_data = data_preparation(config, dataset)
#
# # model loading and initialization
model = CustomLightGCN2(config, train_data.dataset,head_items,tail_items,0.7)
# # 初始化模型
# model = RecModel.LightGCN(config,dataset)
# logger.info(model)
# # trainer loading and initialization
trainer = NewTrainer(config, model)
# trainer = Trainer(config, model)
#
# # model training
best_valid_score, best_valid_result = trainer.fit(train_data, valid_data, show_progress=True)
#
# # model evaluation
test_result = trainer.evaluate(test_data, show_progress=True)
# shh,sht,sth,stt=model.calculate_uninteracted_scores(dataset,head_users,tail_users,head_items,tail_items)
# print("头部用户探索度头部商品"+str(shh)+"，头部用户探索度长尾商品"+str(sht)+"，尾部用户探索度头部商品"+str(sth)+"，尾部用户探索度长尾商品"+str(stt))
# logger.info('best valid result: {}'.format(best_valid_result))
# logger.info('test result: {}'.format(test_result))
