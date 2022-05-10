import json
import os
import sys
from time import time

import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
# from model_resnet import vit_base_patch16_224_in21k as create_model
# from pythonProject1.vision_transforms.model import vit_base16
from model import vit_base_patch16_num10 as create_model
from torch.utils.tensorboard import SummaryWriter

transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
}

batch_size = 16
# 轮数
epochs = 10
# 最优的权重保存的位置
save_path = "./weights/vit_cifar_6test.pth"

# 最后的测试训练增长图展示的地方
tb_writer = SummaryWriter('./log_cifar10_6test')

# path_train = "../../data_set/flower_data/train"
# path_test = "../../data_set/flower_data/val"

# train_data = torchvision.datasets.ImageFolder(root=path_train, transform=transform["train"])
# test_data = torchvision.datasets.ImageFolder(root=path_test, transform=transform["val"])
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=transform["train"], download=False)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=transform["val"], download=False)

train_num = len(train_data)
test_num = len(test_data)
print("训练集个数：{}".format(train_num))
print("测试集个数：{}".format(test_num))


train_downloader = DataLoader(train_data, batch_size, shuffle=True)
test_downloader = DataLoader(test_data, batch_size, shuffle=True)

# print(train_downloader)

flower_list = train_data.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
json_str = json.dumps(cla_dict, indent=4)
with open("class_indices.json", "w") as json_file:
    json_file.write(json_str)

loss_fn = nn.CrossEntropyLoss()
# 使用的是花的数据 为5      cifai10   10分类
net = create_model(num_classes=10, has_logits=None)
# print(net)
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
    net = net.cuda()
# 优化器
learning_rate = 0.003
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# vit_cifar-3.pth
weights_path = "./weights/vit_base_patch16_224_in21k.pth"

print(net.has_logits)
# 加载与预训练参数
# 参数层
if weights_path != "":
    assert os.path.exists(weights_path), "weights file: '{}' not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location= "cuda:0")
    # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] if net.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    print(net.load_state_dict(weights_dict, strict=False))
# net.load_state_dict(weights_dict, strict=False)

freeze_layers = True
if freeze_layers:
    for name, para in net.named_parameters():
        # 除head, pre_logits外，其他权重全部冻结
        if "head" not in name and "pre_logits" not in name:
            para.requires_grad_(False)
        else:
            print("training {}".format(name))
# weights_path = "./weights/vit_cifar_1.pth"
# net.load_state_dict(torch.load(weights_path), strict=False)

best_train_acc = 0.0
best_acc = 0.0

start_time = time()
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    acc_train = 0.0
    sample_num = 0
    train_bar = tqdm(train_downloader, file=sys.stdout)
    for step, data in enumerate(train_bar):  # 这里没有使用enumarate()  取索引引起之前的错误
        img, label = data
        sample_num += img.shape[0]
        # print("img:{}".format(img))
        if torch.cuda.is_available():
            img = img.cuda()
            label = label.cuda()
        outputs = net(img)
        loss = loss_fn(outputs, label)

        predict_y = torch.max(outputs, dim=1)[1]
        acc_train += torch.eq(predict_y, label).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss = running_loss + loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.5f} train_acc:{:.3f}".format(epoch+1, epochs,
                                                                                  running_loss / sample_num,
                                                                                  acc_train / sample_num)

    if acc_train / train_num > best_train_acc:
        best_train_acc = acc_train / train_num

    tb_writer.add_scalar("train_acc", acc_train / train_num, epoch+1)
    tb_writer.add_scalar("train_loss", running_loss / train_num, epoch+1)

    net.eval()
    accurate = 0.0
    with torch.no_grad():
        for val_data in test_downloader:
            val_imgs, val_labels = val_data
            if torch.cuda.is_available():
                val_imgs = val_imgs.cuda()
                val_labels = val_labels.cuda()
            val_outputs = net(val_imgs)
            predict_y = torch.max(val_outputs, dim=1)[1]
            accurate += torch.eq(predict_y, val_labels).sum().item()
        print("[epoch %d] train_loss:%.5f   val_accurate: %.3f" %
              (epoch+1, running_loss / train_num, accurate / test_num))

        if best_acc < accurate / len(test_data):
            best_acc = accurate / test_num
            torch.save(net.state_dict(), save_path)

    tb_writer.add_scalar("val_acc", accurate / test_num, epoch+1)

end_time = time()
print("训练集最好的结果train_best_acc:{}".format(best_train_acc))
print("测试集最好的结果val_best_acc:{}".format(best_acc))

print("花费的时间:{:.5f}".format((end_time-start_time)/60))
tb_writer.close()
print("模型训练完成")
