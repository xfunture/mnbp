#coding=utf-8
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
import time

start_time = time.time()


# 根据常识判断无用的'检查项'table_id，过滤掉无用的table_id
def filter_None(data):
    data = data[data['field_results'] != '']
    data = data[data['field_results'] != '未查']
    return data




# 重复数据的拼接操作
def merge_table(df):
    df['field_results'] = df['field_results'].astype(str)
    if df.shape[0] > 1:
        merge_df = " ".join(list(df['field_results']))
    else:
        merge_df = df['field_results'].values[0]
    return merge_df





# 删除掉一些出现次数低，缺失比例大的字段，保留超过阈值的特征
def remain_feature(df,threshold=0.99):
    num_rows = df.shape[0]
    exclude_col = []
    include_col = []
    print(f'总列数{df.shape[1]}')
    for col in df.columns:
        num_missing= df[col].isnull().sum()
        if num_missing ==0:
            continue
        percent_missing = num_missing/num_rows
        if percent_missing > threshold:
            exclude_col.append(col)
        else:
            include_col.append(col)
    print(f'移除缺失数据列的总数{len(exclude_col)}')
    print(f'剩余数据总列数{len(include_col)}')
    return exclude_col,include_col


#找出所有的中文列，删除掉,返回列表
def del_text(df):
    del_col = []
    save_col = []
    for col in df.columns:
        if is_float_dtype(df[col]):
            save_col.append(col)
        else:
            del_col.append(col)
    print(f'总列数{df.shape[1]}')
    print(f'含有中文的列{len(del_col)}')
    print(f'剩余列数{len(save_col)}')
    return del_col


#清理训练集中的指标
def clean_data(x):
    x = str(x)
    if '+' in x:
        index = x.index('+')
        x = x[0:index]
    if '>' in x:  # > 6.22
        index = x.index('>')
        x = x[index + 1:]
    if len(x.split('.')) > 2:  # 2.2.8
        index = x.rindex('.')
        x = x[:index] + x[index + 1:]
    if str(x).isdigit() == False and len(x) > 4:  # 7.75轻度乳糜
        x = x[0:4]

    if '未查' in x or '弃查' in x:
        x = np.nan

    return x


# end_time = time.time()
# print('程序总共耗时:%d 秒' % int(end_time - start_time))


#将part1 数据和part2数据合并
def merge():
    # 读取数据
    train = pd.read_csv('data/meinian_round1_train_20180408.csv', sep=',')
    test = pd.read_csv('data/meinian_round1_test_a_20180409.csv', sep=',')
    data_part1 = pd.read_csv('data/meinian_round1_data_part1_20180408.txt', sep='$')
    data_part2 = pd.read_csv('data/meinian_round1_data_part2_20180408.txt', sep='$')

    # data_part1和data_part2进行合并，并剔除掉与train、test不相关vid所在的行
    part1_2 = pd.concat([data_part1, data_part2], axis=0)  # {0/'index', 1/'columns'}, default 0
    part1_2 = pd.DataFrame(part1_2).sort_values('vid').reset_index(drop=True)
    vid_set = pd.concat([train['vid'], test['vid']], axis=0)
    vid_set = pd.DataFrame(vid_set).sort_values('vid').reset_index(drop=True)
    part1_2 = part1_2[part1_2['vid'].isin(vid_set['vid'])]

    part1_2 = filter_None(part1_2)

    # 过滤列表，过滤掉不重要的table_id 所在行
    filter_list = ['0203', '0209', '0702', '0703', '0705', '0706', '0709', '0726', '0730', '0731', '3601',
                   '1308', '1316']

    part1_2 = part1_2[~part1_2['table_id'].isin(filter_list)]
    # 数据简单处理
    print(part1_2.shape)
    vid_tabid_group = part1_2.groupby(['vid', 'table_id']).size().reset_index()  #
    # print(vid_tabid_group.head())
    # print(vid_tabid_group.shape)
    #                      vid               table_id  0
    # 0  000330ad1f424114719b7525f400660b     0101     1
    # 1  000330ad1f424114719b7525f400660b     0102     3

    # 重塑index用来去重,区分重复部分和唯一部分
    print('------------------------------去重和组合-----------------------------')
    vid_tabid_group['new_index'] = vid_tabid_group['vid'] + '_' + vid_tabid_group['table_id']
    vid_tabid_group_dup = vid_tabid_group[vid_tabid_group[0] > 1]['new_index']

    # print(vid_tabid_group_dup.head()) #000330ad1f424114719b7525f400660b_0102
    part1_2['new_index'] = part1_2['vid'] + '_' + part1_2['table_id']

    dup_part = part1_2[part1_2['new_index'].isin(list(vid_tabid_group_dup))]
    dup_part = dup_part.sort_values(['vid', 'table_id'])
    unique_part = part1_2[~part1_2['new_index'].isin(list(vid_tabid_group_dup))]

    part1_2_dup = dup_part.groupby(['vid', 'table_id']).apply(merge_table).reset_index()
    part1_2_dup.rename(columns={0: 'field_results'}, inplace=True)
    part1_2_res = pd.concat([part1_2_dup, unique_part[['vid', 'table_id', 'field_results']]])

    table_id_group = part1_2.groupby('table_id').size().sort_values(ascending=False)
    table_id_group.to_csv('temp/part_tabid_size .csv', encoding='utf-8')

    # 行列转换
    print('--------------------------重新组织index和columns---------------------------')
    merge_part1_2 = part1_2_res.pivot(index='vid', values='field_results', columns='table_id')
    print('--------------新的part1_2组合完毕----------')
    print(merge_part1_2.shape)
    merge_part1_2.to_csv('temp/merge_part1_2.csv', encoding='utf-8')
    print(merge_part1_2.head())




def main():
    merge()


if __name__ == "__main__":
    main()
