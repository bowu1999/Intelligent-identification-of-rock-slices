from dataloader import Getdata
from mymodels import RockSlice
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
import json

def train(model, device, train_loader, f_loss, optimizer, epoch):
    model.train()
    train_epoch_loss = []
    correct = 0.
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = f_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_epoch_loss.append(loss.item())
        output = pred.argmax(dim=1)
        correct += output.eq(target.argmax(1)).sum().item()
        if idx % 100 == 0:
            print('Train Epoch: {}, iteration: {}, Loss: {}'.format(epoch, idx, loss.item()))
    acc = correct / len(train_loader.dataset) * 100.
    print('Train Accuracy:{}'.format(acc))
    return train_epoch_loss,acc

def test(model, device, test_loader, f_loss):
    model.eval()
    val_epoch_loss = []
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = f_loss(output, target)
            val_epoch_loss.append(loss.item())
            pred = output.argmax(dim=1)
            correct += pred.eq(target.argmax(1)).sum().item()
    acc = correct / len(test_loader.dataset) * 100.
    print('Test loss: {}, Accuracy:{}'.format(loss.item(), acc))
    return val_epoch_loss,acc

def main():
	
	# 数据读取
	root = "DataSets"
	crop_size=(896,896)
	trainset = Getdata(True,root,crop_size)
	valset = Getdata(False,root,crop_size)
	trainloader = DataLoader(dataset=trainset, batch_size=4, shuffle=True)
	valloader = DataLoader(dataset=valset, batch_size=4, shuffle=True)
	
	# 参数设置
	epochs = 10
	class_num = 20 #种类数
	learning_rate = 0.01
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model = RockSlice(class_num).to(device)
	params = [{'params': filter(lambda p:p.requires_grad, model.get_decoder_params())},
	          {'params': filter(lambda p:p.requires_grad, model.get_backbone_params()), 
	           'lr': learning_rate/10}]
	celoss = nn.CrossEntropyLoss() # 交叉熵损失函数
	adam = torch.optim.Adam(params,
	                lr=learning_rate,
	                betas=(0.9, 0.999),
	                eps=1e-08,
	                weight_decay=0,
	                amsgrad=False)
	
	# 记录训练过程数据
	train_epochs_loss = []
	train_epochs_acc = []
	val_epochs_loss = []
	val_epochs_acc = []

	# 训练
	for epoch in range(epochs):
	    train_epoch_loss,train_epoch_acc = train(model, device, trainloader, celoss, adam, epoch)
	    train_epochs_loss.append(np.average(train_epoch_loss))
	    train_epochs_acc.append(train_epoch_acc)
	    val_epoch_loss,val_epoch_acc = test(model, device, valloader, celoss)
	    val_epochs_loss.append(np.average(val_epoch_loss))
	    val_epochs_acc.append(val_epoch_acc)

	## 保存模型
	torch.save({'model': model.state_dict()}, 'trained_models/01_RockSlice.pth')
	## 读取模型
	# model = RockSlice()
	# state_dict = torch.load('trained_models/01_RockSlice.pth')
	# model.load_state_dict(state_dict['model'])

	# 训练过程参数
	json_path = "training_process_data/01_RockSlice/1_training.json"
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
