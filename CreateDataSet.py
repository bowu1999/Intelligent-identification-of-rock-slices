import os
import random

class CreateDataSets():
    def __init__(self,root,rate=7):
        '''
        root:数据集主目录名
        rate：train和val划分比例，默认：7，即train：val = 7:3
        '''
        assert rate >= 0 and rate <= 10,"rate必须是0-10"
        self.savepath = root
        self.root = os.path.join(root,'Images')
        self.class_first = os.listdir(self.root)
        self.rate = rate
    
    def __get_data_set(self):
        '''
        获取所有的图片路径
        '''
        data_set = []
        class_set = set()
        for first_name in self.class_first:
            for scend_name in os.listdir(os.path.join(self.root,first_name)):
                for data_name in os.listdir(os.path.join(self.root,first_name,scend_name)):
                    data_set.append(os.path.join(first_name,scend_name,data_name))
                    class_set.add(os.path.join(first_name,scend_name))
        return data_set,class_set
    
    def __rand_sort(self,data_set):
        '''
        随机打乱列表
        '''
        random.shuffle(data_set)
        return data_set
    
    def __crop_data_set(self,data_set):
        '''
        将数据集切割成train和val
        '''
        incision_location = (len(data_set) // 10) * self.rate
        return data_set[:incision_location],data_set[incision_location:]
    
    def __save_data(self,data_set,name):
        '''
        将数据保存成txt文件
        '''
        str = '\n'
        f=open(os.path.join(self.savepath,"{}.txt".format(name)),"w")
        f.write(str.join(data_set))
        f.close()
        
    def main(self):
        #获取数据
        data_set,class_set = self.__get_data_set()
        data_set = self.__rand_sort(data_set)
        #获取train_set以及val_set
        train_set,val_set = self.__crop_data_set(data_set)
        #保存
        self.__save_data(train_set,"train")
        self.__save_data(val_set,"val")
        self.__save_data(class_set,"classes")
        
if __name__ == "__main__":
    root = "DataSets"
    mydataset = CreateDataSets(root,8)
    mydataset.main()