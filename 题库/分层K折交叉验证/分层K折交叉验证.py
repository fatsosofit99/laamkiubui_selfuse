import numpy as np

from typing import List, Tuple

def stratified_k_fold_cross_validation(X: List, Y: List, k: int=5, random_seed: int=11, shuffle: bool=False)->Tuple[List, List]:
    #TODO
    
    n = len(X)
    indices = np.arange(n)
    # 1. 处理随机化
    if shuffle:
        np.random.seed(random_seed)
        # 注意：为了保持 X 和 Y 的对应关系，我们只打乱索引
        np.random.shuffle(indices)
    from collections import defaultdict
    # 2. 按类别分组存放索引
    # 这样可以确保我们在每一类内部都能进行均匀的 K 折划分
    label_to_indices = defaultdict(list)
    for idx in indices:
        label = Y[idx]
        label_to_indices[label].append(idx)
    
    # 初始化存储结构：每个 Fold 的验证集和训练集
    # 先初始化 k 个空的测试集列表
    folds_test_indices = [[] for _ in range(k)]
    
    # 3. 核心分层逻辑：在每个类别内部模仿 K-Fold 切分
    for label in sorted(label_to_indices.keys()):
        class_indices = label_to_indices[label]
        n_class = len(class_indices)
        
        base_size = n_class // k
        remainder = n_class % k
        
        start_idx = 0
        for i in range(k):
            current_fold_size = base_size + 1 if i < remainder else base_size
            end_idx = start_idx + current_fold_size
            
            # 将该类别当前折的索引分配给对应的 Fold
            folds_test_indices[i].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx
            
    # 4. 构建最终的训练集和验证集列表
    train_indices_list = []
    test_indices_list = []
    
    all_indices_set = set(indices)
    
    for i in range(k):
        # 当前 Fold 的验证集（转为列表并排序，保持习惯性整洁）
        test_idx = sorted(folds_test_indices[i])
        
        # 训练集 = 总集 - 验证集
        train_idx = sorted(list(all_indices_set - set(test_idx)))
        
        test_indices_list.append(test_idx)
        train_indices_list.append(train_idx)
        
    return train_indices_list, test_indices_list
if __name__ == '__main__':
    # 示例数据集：包含 9 份数据，每一份数据是一张 (3, 224, 224) 的图像
    X = np.random.rand(9, 3, 224, 224).tolist()
    Y = [1, 1, 1, 2, 2, 2, 3, 3, 3]

    # 示例一：划分数据集前不打乱原数据的顺序
    k = 3
    train_indices_list, test_indices_list = stratified_k_fold_cross_validation(X, Y, k)
    
    # 示例二：划分数据集前打乱原数据的顺序
    # k = 3
    # random_seed = 11
    # shuffle = True
    # train_indices_list, test_indices_list = stratified_k_fold_cross_validation(X, Y, k, random_seed, shuffle)
    
    # 打印结果
    for i in range(len(train_indices_list)):
        print(f'Fold {i+1}: {train_indices_list[i]}, {test_indices_list[i]}')
    '''
        Fold 1: [1, 2, 4, 5, 7, 8], [0, 3, 6]
        Fold 2: [0, 2, 3, 5, 6, 8], [1, 4, 7]
        Fold 3: [0, 1, 3, 4, 6, 7], [2, 5, 8]
    '''