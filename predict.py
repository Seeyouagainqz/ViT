import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from pythonProject1.Googlenet.model import GoogLeNet
from model import vit_base_patch16_num10 as create_model
# 数据预处理的设置
# 打开图片
# 图片的预处理， 增加一个batch维度   torch.unsqueeze()
# 模型的加载
# 模型权重参数的加载   net.load_state_dict(torch.load(model_weigths_path), strict=False)
# 分类类别的加载     class_index = json.load(json_file)
# 图片的预测    缩减batch这个维度,  使用torch.softmax()  , torch.argmax()   得到这个点最大的索引
data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

model_weigths_path = "./weights/vit-base.pth"
img_path = "./002.PNG"
img = Image.open(img_path).convert('RGB')
plt.imshow(img)

img = data_transform(img)
img = torch.unsqueeze(img, dim=0)

net = create_model(num_classes=10, has_logits=False)
if torch.cuda.is_available():
    net = net.cuda()
    img = img.cuda()
    print(1)

try:
    json_file = open("./class_indices.json", "r")
    class_index = json.load(json_file)    # 解码成字典的格式   ,里边存储的是相应的类别的名字
except Exception as e:
    print(e)   # 输出错误的原因
    exit(-1)


missing_keys, unexpected_keys = net.load_state_dict(torch.load(model_weigths_path), strict=False)   #
# 这个unexpected_keys  存储的合适的参数矩阵的权重
# 为True会精准的匹配载入模型和权重模型

net.eval()
with torch.no_grad():
    outputs = net(img).cpu()
    outputs = torch.squeeze(outputs)
    predict = torch.softmax(outputs, dim=0)
    predict_y = torch.argmax(predict).numpy()

# print("预测花的类别：{}，预测的概率{}。".format(class_index[str(predict_y)], predict[predict_y].item()))
print("预测的类别：{}， 预测的概率{}。".format(class_index[str(predict_y)], predict[predict_y].item()))

print("预测结束")