import numpy as np
from typing import Tuple

def compute_l1_importance(conv_weights: np.ndarray) -> np.ndarray:
    #TODO
    importance_scores=np.sum(np.abs(conv_weights),axis=(1,2,3))
    return importance_scores

def prune_conv_layer(conv1_weights: np.ndarray, conv1_bias: np.ndarray, keep_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    importance_scores = compute_l1_importance(conv1_weights)
    #TODO
    importance_scores=compute_l1_importance(conv1_weights)
    total_out_channels=conv1_weights.shape[0]
    num_keep=int(total_out_channels*keep_ratio)
    top_indices=np.argsort(importance_scores)[-num_keep:]
    keep_indices=np.sort(top_indices)

    pruned_weights=conv1_weights[keep_indices,:,:,:]
    pruned_bias=conv1_bias[keep_indices]

    return pruned_weights,pruned_bias,keep_indices


def adjust_next_layer(conv2_weights: np.ndarray, keep_indices: np.ndarray) -> np.ndarray:
    #TODO
    adjusted_weights=conv2_weights[:,keep_indices,:,:]
    return adjusted_weights

if __name__ == "__main__":
    np.random.seed(42)
    conv1_w = np.random.randn(5, 3, 3, 3)
    conv1_b = np.random.randn(5)
    conv2_w = np.random.randn(4, 5, 3, 3)

    print("== 剪枝前 ==")
    print(f"Conv1 权重 shape: {conv1_w.shape}")
    print(f"Conv2 权重 shape: {conv2_w.shape}")

    pruned_w, pruned_b, keep_ids = prune_conv_layer(conv1_w, conv1_b, keep_ratio=0.6)
    adjusted_conv2 = adjust_next_layer(conv2_w, keep_ids)

    print("\n== 剪枝后 ==")
    print(f"Conv1 权重 shape: {pruned_w.shape}")
    print(f"Conv2 权重 shape: {adjusted_conv2.shape}")
    print(f"保留的输出通道索引: {keep_ids.tolist()}")