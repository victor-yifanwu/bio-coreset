import pickle
import torch
import numpy as np

def monotonic_sampling_numpy(data, sample_num):
    """
    Select the indices of the smallest `sample_num` values from the array.

    Args:
        data (np.ndarray): 1D numpy array of scores.
        sample_num (int): Number of smallest elements to select.

    Returns:
        selected_indices (np.ndarray): Array of selected indices (int64).
    """
    if sample_num > len(data):
        raise ValueError(f"sample_num {sample_num} is larger than data length {len(data)}")

    # 获取按值从小到大排序后的索引
    sorted_idx = np.argsort(data)

    # 取前 sample_num 个最小值的索引
    selected_indices = sorted_idx[:sample_num]

    return selected_indices

def stratified_sampling_numpy(data, sample_num, stratas=50, seed=42, exclude_lowest_ratio=0.0):
    """
    Perform stratified sampling on a 1D score array.

    Args:
        data (np.ndarray): 1D numpy array of scores.
        sample_num (int): Total number of samples to select.
        stratas (int): Number of bins to stratify into.
        seed (int): Random seed for reproducibility.
        exclude_lowest_ratio (float): Fraction of lowest scores to exclude (e.g. 0.1 = drop lowest 10%).

    Returns:
        selected_indices (np.ndarray): Array of selected sample indices.
    """
    
    np.random.seed(seed)
    
    all_indices = np.arange(len(data))

    if exclude_lowest_ratio > 0:
        # Step 1: 计算排除阈值
        cutoff_index = int(len(data) * exclude_lowest_ratio)
        sorted_idx = np.argsort(data)  # 从小到大排序
        mask = np.ones(len(data), dtype=bool)
        mask[sorted_idx[:cutoff_index]] = False

        data = data[mask]
        all_indices = all_indices[mask]

        print(f"Excluded lowest {exclude_lowest_ratio*100:.1f}% scores -> retained {len(data)} samples.")

    # Step 2: 分层边界计算
    min_score = data.min()
    max_score = data.max() * 1.0001
    step = (max_score - min_score) / stratas

    def bin_range(k):
        return min_score + k * step, min_score + (k + 1) * step

    # Step 3: 统计各分层样本数量
    strata_num = np.zeros(stratas, dtype=int)
    for i in range(stratas):
        start, end = bin_range(i)
        mask = (data >= start) & (data < end)
        strata_num[i] = mask.sum()
    print(strata_num)

    # Step 4: 预算分配
    sorted_idx = np.argsort(strata_num)
    budgets = np.zeros(stratas, dtype=int)
    remaining = sample_num

    for i, strata_i in enumerate(sorted_idx):
        rest_bins = stratas - i
        avg = remaining // rest_bins
        this_budget = min(strata_num[strata_i], avg)
        budgets[strata_i] = this_budget
        remaining -= this_budget

    # Step 5: 分层随机采样
    selected_indices = []
    for i in range(stratas):
        start, end = bin_range(i)
        mask = (data >= start) & (data < end)
        candidates = all_indices[mask]
        if budgets[i] > 0 and len(candidates) > 0:
            sampled = np.random.choice(candidates, size=budgets[i], replace=False)
            selected_indices.extend(sampled.tolist())

    return np.array(selected_indices)

data = np.load("./result/inf_scores.npy")  # shape: (20000000,)

# --- 1. Stratified sampling ---
selected_idx_strata = stratified_sampling_numpy(
    data,
    sample_num=200000,        # 目标数量
    stratas=50,
    exclude_lowest_ratio=0.1  # 去掉最低 10%
)
np.save("./result/ccs200k_whole_idx.npy", selected_idx_strata)
print(f"[Stratified] Selected {len(selected_idx_strata)} samples -> saved to ./result/ccs200k_whole_idx.npy")

# --- 2. Top-k smallest ---
selected_idx_topk = monotonic_sampling_numpy(
    data,
    sample_num=200000         # 目标数量
)
np.save("./result/top200k_whole_idx.npy", selected_idx_topk)
print(f"[Top-k] Selected {len(selected_idx_topk)} samples -> saved to ./result/top200k_whole_idx.npy")
