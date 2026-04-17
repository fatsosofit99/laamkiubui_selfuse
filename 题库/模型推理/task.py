import os
import json
import numpy as np
import random
from torchvision import models, transforms
import torch
from torch import nn
from typing import Tuple, List
from torch import Tensor

def set_random_seed(random_seed: int=42):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

def load_model(model: nn.Module, weight_path: str)->nn.Module:
    state_dick = torch.load(weight_path)
    model.load_state_dict(state_dick)
    return model
    #TODO


def load_data(data_path: str, transform: transforms.Compose)->Tensor:
    from PIL import Image
    image_tensor=[]
    for f in sorted(os.listdir(data_path)):
        path = os.path.join(data_path,f)
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        image_tensor.append(img_tensor)
    bitch_tensor = torch.stack(image_tensor)
    return bitch_tensor
    #TODO


def inference(model: nn.Module, images: Tensor)->Tensor:
    model.eval()
    with torch.no_grad():
        output=model(images)
        pred = torch.argmax(output,dim=1)
    return pred
    #TODO


def cal_metrics(y_true: List, y_pred: List)->Tuple[float, float, float, float]:
    yt = np.array(y_true)
    yp = np.array(y_pred)
    accuracy = np.mean(yt==yp)

    classes = np.unique(np.concatenate(((yt,yp))))
    num_classes = len(classes)
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for c in classes:
        # True Positive (真正例)：预测为 c，实际也为 c
        tp = np.sum((yp == c) & (yt == c))
        # False Positive (假正例)：预测为 c，但实际不是 c
        fp = np.sum((yp == c) & (yt != c))
        # False Negative (假负例)：实际为 c，但预测不是 c
        fn = np.sum((yp != c) & (yt == c))
    # 计算当前类别的 Precision, Recall 和 F1 (等同于 zero_division=0)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
        
        # 累加到总和中
        macro_precision += p
        macro_recall += r
        macro_f1 += f1
        
    # 3. 计算宏平均 (Macro Average): 将所有类的指标求算术平均
    macro_precision /= num_classes
    macro_recall /= num_classes
    macro_f1 /= num_classes
    
    return float(accuracy), float(macro_precision), float(macro_recall), float(macro_f1)
    #TODO
    

def test(model: nn.Module, weight_path: str, data_path: str, transform: transforms.Compose, y_true: List)->Tuple[List, float, float, float, float]:
    # 模型加载
    model = load_model(model, weight_path)
    # 数据加载
    images = load_data(data_path, transform)
    
    # 模型推理
    model.eval()
    pred = inference(model, images)

    # 计算评价指标：ACC、Precision、Recall、F1-Score
    y_pred = pred.numpy().tolist()
    acc, precision, recall, f1_score = cal_metrics(y_true, y_pred)
    
    return y_pred, acc, precision, recall, f1_score

if __name__ == '__main__':
    # 设置随机种子
    set_random_seed(42)

    model = models.resnet18()
    root_path = '/home/project'
    weight_path = os.path.join(root_path, 'resnet18-5c106cde.pth')
    data_path = os.path.join(root_path, 'data')
    transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
    y_true = []
    label_json_path = os.path.join(root_path, 'labels.json')
    with open(label_json_path, 'r') as json_file:
        y_true = json.load(json_file)["label"]
    
    y_pred, acc, precision, recall, f1_score = test(model, weight_path, data_path, transform, y_true)
    print(len(y_pred) == len(y_true)) # True
    print(f'Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}')
    # Accuracy: 0.7936507936507936, Precision: 0.07142857142857142, Recall: 0.056689342403628114, F1 Score: 0.06321112515802782