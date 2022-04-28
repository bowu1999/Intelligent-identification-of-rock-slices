from statistics import mode
from dataloader import Getdata
from mymodels import *
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
from tqdm import tqdm


def main():
	# 参数设置
	# 训练次数
	epochs = 300
	# 种类数
	class_num = 17
	# 训练的初始识别率，采用poly方式修改
	learning_rate = 0.01
	# batch大小
	batch_size = 32
	# models = ["rockslice","resnet50","resnet101","vgg","resnext50"]
	model_name = "rockslice"
	# 数据读取
	root = "DataSets_{}".format(class_num)
	crop_size=(896,896)
	train_mid_set = Getdata(True,root,crop_size,"middle")
	train_max_set = Getdata(True,root,crop_size)
	val_mid_set = Getdata(False,root,crop_size,"middle")
	val_max_set = Getdata(False,root,crop_size)
	trainloader_mid = DataLoader(dataset=train_mid_set, batch_size=batch_size, shuffle=True)
	trainloader_max = DataLoader(dataset=train_max_set, batch_size=batch_size, shuffle=True)
	valloader_mid = DataLoader(dataset=val_mid_set, batch_size=batch_size, shuffle=True)
	valloader_max = DataLoader(dataset=val_max_set, batch_size=batch_size, shuffle=True)
	# GPU
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# 模型选择
	if model_name == "rockslice":
		model = RockSlice(class_num)
	if model_name == "resnet50":
		model = ResNet50(class_num)
	if model_name == "resnet101":
		model = ResNet101(class_num)
	if model_name == "vgg":
		model = VGG(class_num)
	if model_name == "resnext50":
		model = ResNeXt50(class_num)
	# 读取预训练模型模型
	# state_dict = torch.load('trained_models/2_classes_is17_RockSlice.pth')
	# model.load_state_dict(state_dict['model'])
	
	# 需训练的参数
	params = [{'params': filter(lambda p:p.requires_grad, model.get_decoder_params())},
	          {'params': filter(lambda p:p.requires_grad, model.get_backbone_params()), 
	           'lr': learning_rate/10}]
	model = model.to(device)
	# 交叉熵损失函数
	celoss = nn.CrossEntropyLoss()
	# 优化器
	optimizer = torch.optim.SGD(params,lr=learning_rate,momentum=0.9,dampening=0.5,weight_decay=0.01,nesterov=False)
	# 记录训练过程数据
	train_epochs_loss = []
	train_epochs_acc = []
	val_epochs_loss = []
	val_epochs_acc = []
	# 当前训练学习率
	now_lr = learning_rate
	print("共分类：",class_num)
	# 训练
	for epoch in range(epochs):
		now_lr = adjust_learning_rate_poly(optimizer, epoch % 100, 100, learning_rate, 0.5)
		print("Now learning_rate:",now_lr)
		train_epoch_loss,train_epoch_acc = train(model, device, trainloader_mid, trainloader_max, celoss, optimizer, epoch)
		train_epochs_loss.append(np.average(train_epoch_loss))
		train_epochs_acc.append(train_epoch_acc)
		val_epoch_loss,val_epoch_acc = test(model, device, valloader_mid, valloader_max, celoss,epoch)
		val_epochs_loss.append(np.average(val_epoch_loss))
		val_epochs_acc.append(val_epoch_acc)
		if epoch != 0 and epoch % 10 == 0:
			save(model, model_name, class_num, train_epochs_loss, train_epochs_acc, val_epochs_loss, val_epochs_acc, epoch)
			print("模型保存一次")
		if epoch == 100:
			# 一百轮训练之后冻结底层特征提取的参数
			freeze(model)
			params = [{'params': filter(lambda p:p.requires_grad, model.get_decoder_params())},
	          {'params': filter(lambda p:p.requires_grad, model.get_backbone_params()), 
	           'lr': learning_rate/10}]
			optimizer = torch.optim.SGD(params,lr=learning_rate,momentum=0.9,dampening=0.5,weight_decay=0.01,nesterov=False)
	save(model, model_name, class_num, train_epochs_loss, train_epochs_acc, val_epochs_loss, val_epochs_acc, epochs)
	print("训练过程已保存")

def freeze(model):
	# 冻结预训练参数
	for name, param in model.named_parameters():
		if "backbone" in name:
			if "layer3" in name:
				break
			param.requires_grad = False

def accuracy(output, target, topk=(1,)):
    """计算指定 k 值的精度"""
    tarlist = target.argmax(1)
    res = []
    for k in topk:
        corr = 0
        _, pred = output.topk(k, 1, True, True)  # 返回最大的k个结果（按最大到小排序）
        for tar,outset in zip(tarlist,pred):
            if tar in outset:
                corr += 1
        res.append(corr)        
    return res

def train(model, device, trainloader_mid, trainloader_max, f_loss, optimizer, epoch):
	model.train()
	#model.freeze_bn()
	train_epoch_loss = []
	correct = 0.
	for data_mid,data_max in tqdm(zip(enumerate(trainloader_mid),enumerate(trainloader_max))):
		idx = data_mid[0]
		img_mid = data_mid[1][0].to(torch.float32).to(device)
		img_max = data_max[1][0].to(torch.float32).to(device)
		target = data_max[1][1].to(torch.float32).to(device)
		x = (img_mid,img_max)
		output = model(*x)
		loss = f_loss(output, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_epoch_loss.append(loss.item())
		pred = output.argmax(dim=1)
		correct += pred.eq(target.argmax(1)).sum().item()
		if idx % 100 == 0:
			print('Train Epoch: {}, iteration: {}, Loss: {}'.format(epoch, idx, np.average(train_epoch_loss)))
	print('Train Epoch: {}, iteration: {}, Loss: {}'.format(epoch, idx, np.average(train_epoch_loss)))
	acc = correct / len(trainloader_max.dataset) * 100.
	print('Train Accuracy:{}'.format(acc))
	return train_epoch_loss,acc

def test(model, device, testloader_mid, testloader_max, f_loss,epoch):
	model.eval()
	val_epoch_loss = []
	correct = 0.
	correctk = 0.
	with torch.no_grad():
		for data_mid,data_max in tqdm(zip(enumerate(testloader_mid),enumerate(testloader_max))):
			idx = data_mid[0]
			img_mid = data_mid[1][0].to(torch.float32).to(device)
			img_max = data_max[1][0].to(torch.float32).to(device)
			target = data_max[1][1].to(torch.float32).to(device)
			x = (img_mid,img_max)
			output = model(*x)
			loss = f_loss(output, target)
			val_epoch_loss.append(loss.item())
			a,ak = accuracy(output, target, topk=(1,3))
			correct += a
			correctk += ak
			if idx % 100 == 0:
				print('Test Epoch: {}, iteration: {}, Loss: {}'.format(epoch, idx, np.average(val_epoch_loss)))
	print('Test Epoch: {}, iteration: {}, Loss: {}'.format(epoch, idx, np.average(val_epoch_loss)))
	acc = correct / len(testloader_max.dataset) * 100.
	acck = correctk / len(testloader_max.dataset) * 100.
	print('Test Accuracy top1:',acc,',top3:',acck)
	return val_epoch_loss,acc

# 学习率的poly策略
def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
	lr = base_lr * (1-epoch/num_epochs)**power
	scal = 1
	for param_group in optimizer.param_groups:
		new_lr = lr/scal
		if new_lr < 0.000001:
			new_lr = 0.000001
		param_group['lr'] = new_lr
		scal *= 10
	return lr	

# 保存模型及过程数据
def save(model, model_name, class_num, train_epochs_loss, train_epochs_acc, val_epochs_loss, val_epochs_acc, epoch):
	## 保存模型
	torch.save({'model': model.state_dict()}, 'trained_models/实验1_classes_is{}_model_is_{}.pth'.format(class_num, model_name))
	# 训练过程参数
	json_path = "training_process_data/实验1_classes_is{}_model_is_number{}s_training.json".format(class_num, model_name, epoch)
	save_params = {
	    "train_epochs_loss":train_epochs_loss,
	    "train_epochs_acc":train_epochs_acc,
	    "val_epochs_loss":val_epochs_loss,
	    "val_epochs_acc":val_epochs_acc}
	# 保存
	jsObj = json.dumps(save_params)  
	fileObject = open(json_path, 'w')  
	fileObject.write(jsObj)  
	fileObject.close()

	# 读取
	# f = open(json_path,'r',encoding='utf-8')
	# training_process_params = json.load(f)

if __name__ == '__main__':
	main()
