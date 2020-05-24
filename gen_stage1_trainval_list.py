#encoding=utf-8
'''
对于stage1，采用在线数据增强，因此只需要生成训练列表
'''
import os
import numpy as np
import random
import cv2
import torchvision.transforms
from dataset.data_aug import *
def remove_invalid_image(lines):
    images = []
    for line in lines:
        line_ = line.split()
        name = line_[0]
        rect = list(map(int, list(map(float, line_[1:5]))))
        landm = list(map(float, line_[5:]))
        #去掉坐标越界的人脸
        rect = np.array(rect)
        landm = np.array(landm)
        if (rect >= 0).all()==False or (landm >= 0).all()==False:
            continue
        if os.path.isfile(name):
            images.append(line)
    return images

def load_metadata(folder_list):
    tmp_lines = []
    for folder_name in folder_list:
        label_file = os.path.join(os.path.abspath('.'),"data",folder_name,"label.txt")
        with open(label_file) as f:
            lines = f.readlines()
        tmp_lines=[os.path.join(os.path.abspath('.'),"data",folder_name,line) for line in lines]
    res_lines = remove_invalid_image(tmp_lines)
    return res_lines

def load_truth(lines):
    truth = {}
    for line in lines:
        line = line.strip().split()
        name = line[0]
        if name not in truth:
            truth[name] = []
        rect = list(map(int, list(map(float, line[1:5]))))
        x = list(map(float, line[5::2]))
        y = list(map(float, line[6::2]))
        landmarks = list(zip(x, y))
        truth[name].append((rect, landmarks))
    return truth

def debug():
    folder_list = ["I", "II"]
    lines = load_metadata(folder_list)
    bad_list = [
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\007151.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\004761.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\007194.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\006597.jpg'
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\005694.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\004295.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\004668.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\005977.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\006851.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\005075.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\004335.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\006656.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\005799.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\008017.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\004799.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\008649.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\004993.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\009560.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\004031.jpg'
    ]
    truths = load_truth(lines)
    tran_tsf = torchvision.transforms.Compose([
        Resize(),
        Rotate(prob=0.5),
        # Distort(prob=0.2),
        Mirror(prob=0.5),
        Normalize()]  # convert to torch type: NxCxHxW
    )
    for key in truths:
        if key == bad_list[-2]:
            continue
        values = truths[key]
        img = cv2.imread(key)
        img_height, img_width, _ = img.shape
        for v in values:
            rect = v[0]
            landm = v[1:]
            landmarks = np.squeeze(np.array(landm))
            xy_min = np.min(landmarks, axis=0, keepdims=True).astype(np.int32)
            # 求原始坐标的最大值（原始最右上角）
            xy_max = np.max(landmarks, axis=0, keepdims=True).astype(np.int32)
            # 求landmark的宽度和高度
            wh = xy_max - xy_min + 1
            # 求landmark的中心点(center_x,center_y)
            center = (xy_min + wh / 2).astype(np.int32)
            # 扩大ROI 0.2倍(以最长的边 * 1.2作为boxsize)
            ratio = random.random() + 1
            boxsize = int(np.max(wh) * ratio)
            # 求取扩展后的最左上点
            tl = center - boxsize // 2
            # 扩展后最左上点
            x1, y1 = tl[0, 0], tl[0, 1]
            # 扩展后最右下点
            br = tl + boxsize
            x2, y2 = br[0, 0], br[0, 1]
            # 判断是否超出原始图像
            dx = max(0, -x1)
            dy = max(0, -y1)
            # 避免坐标出现负值
            x1 = max(0, x1)
            y1 = max(0, y1)
            # 判断是否超出原始图像
            edx = max(0, x2 - img_width)
            edy = max(0, y2 - img_height)
            # 限制bbox不超过原图
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            image = img[y1:y2, x1:x2]
            landmarks[:, 0] -= x1  # 将x坐标与crop之后的image对齐
            landmarks[:, 1] -= y1  # 将y坐标与crop之后的image对齐
            sample = {'image': image, 'landmarks': landmarks, 'path': ''}
            sample = tran_tsf(sample)
            image = sample["image"]
            landmarks = sample["landmarks"]
            landmarks = landmarks.reshape(-1, 2) * 112
            # cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)
            for x, y in landmarks.tolist():
                cv2.circle(image, (int(x), int(y)), 1, (0, 0, 255), -1)
            cv2.imshow("image", image)
            cv2.waitKey()
        cv2.imshow("ori", img)
        key = cv2.waitKey(0)


if __name__=="__main__":
    folder_list = ["I","II"]
    lines = load_metadata(folder_list)
    random.shuffle(lines) #打乱顺序
    train_num = int(0.9*len(lines))
    train_list = lines[:train_num]
    val_list = lines[train_num:]
    process_lists = [train_list,val_list]
    label_files = ["stage1_train_list.txt","stage1_val_list.txt"]
    ignore=['D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\007151.jpg',
        'D:\\projects\\python\\Homework\\lesson6\\project2\\data\\II\\007194.jpg']
    for idx,process_list in enumerate(process_lists):
        label_file = open(label_files[idx],"w")
        for l in process_list:
            if l[0] in ignore:
                continue
            label_file.write(l)
        label_file.close()





