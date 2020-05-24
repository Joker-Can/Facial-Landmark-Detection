#encoding=utf-8
'''
对于stage3,因为涉及到负样本的采样，因此选择离线生成的方式
'''
import cv2
import os
import numpy as np
from dataset.data_aug import *
import torchvision.transforms as transforms
def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    # box = (x1, y1, x2, y2)
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # abtain the offset of the interception of union between crop_box and gt_box
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr

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

def gen_neg(img,neg_num_thresh,neg_save_dir,target_size=(112,112),iou=0.1):
    neg_num =0
    global p_idx, n_idx
    while neg_num < neg_num_thresh:
        mean_box = np.mean(boxes,axis=0,keepdims=True)
        x1,y1,x2,y2 = mean_box[0]
        mean_w = x2-x1
        mean_h = y2-y1
        size = np.random.randint(int(min(mean_w, mean_h) * 0.8), max(mean_w, mean_h))
        size = min(size,min(width-size,height-size))
        nx = np.random.randint(0, width - size)
        ny = np.random.randint(0, height - size)

        crop_box = np.array([nx, ny, nx + size, ny + size])
        Iou = IoU(crop_box, boxes)
        cropped_im = img[ny: ny + size, nx: nx + size, :]

        resized_im = cv2.resize(cropped_im, target_size, interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) <= iou:
            # Iou with all gts must below 0.3
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1
    return n_idx

def gen_pos(img,img_path,boxes,landmarks,transformer,pos_num_thresh,pos_save_dir,neg_save_dir,target_size=(112,112),iou=0.65):
    global p_idx, n_idx
    for idx,box in enumerate(boxes):
        landmarksB = landmarks[idx,:]  # 转成x,y格式，便于后面操作
        landmarksB = landmarksB.reshape(-1,2)
        # box (x_left, y_top, x_right, y_bottom)
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        # generate negative examples that have overlap with gt
        for i in range(5):
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = np.random.randint(max(-size, -x1), w)
            delta_y = np.random.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)

            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resized_im = cv2.resize(cropped_im, target_size, interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.2:
                # Iou with all gts must below 0.2
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

        # generate positive examples
        i = 0
        while i<pos_num_thresh:
            img_height, img_width = img.shape[:2]
            x1, y1, x2, y2, _, _ = expand_roi(img_width, img_height, landmarksB)  # 得到在原图上扩充后的bbox
            # 不会超出原图（截取bbox的图像数据）
            imgT = img[y1:y2, x1:x2]
            landmarksT = landmarksB.copy()
            landmarksT[:, 0] -= x1  # 将x坐标与crop之后的image对齐
            landmarksT[:, 1] -= y1  # 将y坐标与crop之后的image对齐
            sample = {'image': imgT, 'landmarks': landmarksT,'path': img_path,'ori_landmark': landmarksB}

            if transformer is not None:
                sample = transformer(sample)
            if np.mean(sample['image'])<50:
                continue
            i+=1
            # for idx, (x, y) in enumerate(sample['landmarks']):
            #     cv2.circle(sample["image"], (int(x), int(y)), 1, (0, 0, 255), -1)
            # cv2.imshow("xx",sample["image"])
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
            landm = sample['landmarks']
            f1.write(save_file+" 1")
            for (x,y) in landm:
                f1.write(' %.2f %.2f ' % (x,y))
            f1.write("\n")
            cv2.imwrite(save_file, sample["image"])
            p_idx += 1

    return n_idx,p_idx

anno_file = "train_list.txt"
root_dir = 'D:/projects/python/Homework/lesson6/project2/'
pos_save_dir = root_dir+"stage3_train_data/train/positive/"
neg_save_dir = root_dir+"stage3_train_data/train/negative/"

# store labels of positive, negative images
f1 = open(root_dir+'stage3_train_data/train/pos.txt', 'w')
f2 = open(root_dir+'stage3_train_data/train/neg.txt', 'w')

# anno_file: store labels of the wider face training data
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)
tran_tsf = transforms.Compose([
        Resize(),
        Rotate(prob=0.8),
        Mirror(prob=0.2),
        Distort(prob=0.2)
      ]
    )

truths = load_truth(annotations)
global p_idx,n_idx
p_idx = 0 # positive
n_idx = 0 # negative
idx = 0
box_idx = 0
neg_num = 100
pos_num = 80

for key in truths:
    img = cv2.imread(key)
    gts = truths[key]
    bboxs = []
    landmarks = []
    for gt in gts:
       bboxs.append(gt[0])
       landmarks.append(gt[1])
    boxes = np.array(bboxs).reshape(-1,4)
    landms = np.array(landmarks).reshape(-1,42)
    idx += 1
    height, width = img.shape[:2]
    gen_pos(img,key,boxes,landms,tran_tsf,pos_num,pos_save_dir,neg_save_dir,target_size=(112,112),iou=0.65)
    gen_neg(img,neg_num, neg_save_dir,target_size=(112, 112),iou=0.1)
    print("%s images done, pos:%s neg: %s" % (idx, p_idx, n_idx))
f1.close()
f2.close()

#生成train val list
root_dir = 'D:/projects/python/Homework/lesson6/project2/'
fp = open(root_dir+'stage3_train_data/train/pos.txt', 'r')
fn = open(root_dir+'stage3_train_data/train/neg.txt', 'r')
pos = fp.readlines()
neg = fn.readlines()
total = pos+neg
random.shuffle(total) #打乱顺序
train_num = int(0.9*len(total))
train_list = total[:train_num]
val_list = total[train_num:]
process_lists = [train_list,val_list]
label_files = ["stage3_train_list.txt","stage3_val_list.txt"]
for idx, process_list in enumerate(process_lists):
    label_file = open(label_files[idx], "w")
    for l in process_list:
        label_file.write(l)
    label_file.close()
fp.close()
fn.close()