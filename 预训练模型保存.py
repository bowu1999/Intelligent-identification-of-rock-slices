import torch
import collections
from mymodels import RockSlice

# resnet50模型参数获取
resnet50_path = "trained_models/resnet50.pth"
resnet_dict = torch.load("trained_models/resnet50.pth")

model = RockSlice(17)
model_keys = [*model.state_dict().keys()][:144]
pretrained_dict = collections.OrderedDict()
pretrained_keys = ["conv1","bn1","layer1","layer2"]
num = 0
for k,v in resnet_dict["model"].items():
        if k[:k.find('.')] in pretrained_keys:
            pretrained_dict[model_keys[num]] = v
            num += 1
for k,v in model.state_dict().items():
    if k[:k.find('.')] not in pretrained_keys:
        pretrained_dict[k] = v

# 预训练模型保存
save_path = "trained_models/pretrained.pth"
torch.save(pretrained_dict, save_path) 