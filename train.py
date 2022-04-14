from dataloader import Getdata
from mymodels import RockSlice03
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
import json

def train(model, device, trainloader_mid, trainloader_max, f_loss, optimizer, epoch):
	model.train()
	train_epoch_loss = []
	correct = 0.
	for data_mid,data_max in zip(enumerate(trainloader_mid),enumerate(trainloader_max)):
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
	acc = correct / len(trainloader_max.dataset) * 100.
	print('Train Accuracy:{}'.format(acc))
	return train_epoch_loss,acc

def test(model, device, testloader_mid, testloader_max, f_loss,epoch):
	model.eval()
	val_epoch_loss = []
	correct = 0.
	with torch.no_grad():
		for data_mid,data_max in zip(enumerate(testloader_mid),enumerate(testloader_max)):
			idx = data_mid[0]
			img_mid = data_mid[1][0].to(torch.float32).to(device)
			img_max = data_max[1][0].to(torch.float32).to(device)
			target = data_max[1][1].to(torch.float32).to(device)
			x = (img_mid,img_max)
			output = model(*x)
			loss = f_loss(output, target)
			val_epoch_loss.append(loss.item())
			pred = output.argmax(dim=1)
			correct += pred.eq(target.argmax(1)).sum().item()
			if idx % 100 == 0:
				print('Test Epoch: {}, iteration: {}, Loss: {}'.format(epoch, idx, np.average(val_epoch_loss)))
	acc = correct / len(testloader_max.dataset) * 100.
	print('Test Accuracy:',acc)
	return val_epoch_loss,acc

def main():
	
	# 数据读取
	root = "DataSets"
	crop_size=(896,896)
	batch_size = 32
	train_mid_set = Getdata(True,root,crop_size,"middle")
	train_max_set = Getdata(True,root,crop_size)
	val_mid_set = Getdata(False,root,crop_size,"middle")
	val_max_set = Getdata(False,root,crop_size)
	trainloader_mid = DataLoader(dataset=train_mid_set, batch_size=batch_size, shuffle=True)
	trainloader_max = DataLoader(dataset=train_max_set, batch_size=batch_size, shuffle=True)
	valloader_mid = DataLoader(dataset=val_mid_set, batch_size=batch_size, shuffle=True)
	valloader_max = DataLoader(dataset=val_max_set, batch_size=batch_size, shuffle=True)
	
	# 参数设置
	epochs = 100
	class_num = 20 #种类数
	learning_rate = 0.005
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = RockSlice03(class_num)
	params = [{'params': filter(lambda p:p.requires_grad, model.get_decoder_params())},
	          {'params': filter(lambda p:p.requires_grad, model.get_backbone_params()), 
	           'lr': learning_rate*2}]
	# 冻结backbone的参数
	for name, param in model.named_parameters():
		if "backbone" in name:
			if "layer4" in name:
				break
			param.requires_grad = False
	# params = [{'params': filter(lambda p:p.requires_grad, model.get_decoder_params())}]
	celoss = nn.CrossEntropyLoss() # 交叉熵损失函数
	adam = torch.optim.Adam(params,
	                lr=learning_rate,
	                betas=(0.9, 0.999),
	                eps=1e-08,
	                weight_decay=0.05,
	                amsgrad=False)
	## 读取模型
	# state_dict = torch.load('trained_models/01_RockSlice.pth')
	# model.load_state_dict(state_dict['model'])
	model = model.to(device)
	# 记录训练过程数据
	train_epochs_loss = []
	train_epochs_acc = []
	val_epochs_loss = []
	val_epochs_acc = []

	# 训练
	for epoch in range(epochs):
		train_epoch_loss,train_epoch_acc = train(model, device, trainloader_mid, trainloader_max, celoss, adam, epoch)
		train_epochs_loss.append(np.average(train_epoch_loss))
		train_epochs_acc.append(train_epoch_acc)
		val_epoch_loss,val_epoch_acc = test(model, device, valloader_mid, valloader_max, celoss,epoch)
		val_epochs_loss.append(np.average(val_epoch_loss))
		val_epochs_acc.append(val_epoch_acc)
	save(model,train_epochs_loss,train_epochs_acc,val_epochs_loss,val_epochs_acc)
	print("训练过程已保存")

def save(model,train_epochs_loss,train_epochs_acc,val_epochs_loss,val_epochs_acc):
	## 保存模型
	torch.save({'model': model.state_dict()}, 'trained_models/01_RockSlice02.pth')
	# 训练过程参数
	json_path = "training_process_data/01_RockSlice02/0_training.json"
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
