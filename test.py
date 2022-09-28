

import os
import sys
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        
        "test": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

#修改文件路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    data_root = os.path.join(data_root, "jiaopp","flower_classfication_resnet")
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # image_path = os.path.join(data_root, "flower_photos") 
    # image_path = r'/data6/jiaopp/flower_classfication_resnet/data_set/flower_data/flower_photos/'
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                         transform=data_transform["test"])
    test_num = len(test_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = test_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 8
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)


    print("using {} images for testing.".format(test_num))
    
    net = resnet34(num_classes=3).to(device)
     # load model weights  加载模型参数
    weights_path = "./resNet34.pth"
    assert os.path.exists(weights_path), f"file: '{weights_path}' dose not exist."
    net.load_state_dict(torch.load(weights_path, map_location=device))
  
    # prediction
    net.eval()
    batch_size = 8  # 每次预测时将多少张图片打包成一个batch
                                                         #  loss)
    classes = ('fire', 'other', 'person')
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    # testing 
    net.eval()

    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, test_labels = test_data
            outputs = net(test_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

            # val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
            #                                            epochs)

            
            for label, prediction in zip(test_labels, predict_y):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))
    test_accurate = acc / test_num
    print('test_accuracy: %.3f' % (test_accurate))



if __name__ == '__main__':
    main()
