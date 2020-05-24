# encoding=utf-8
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from dataset.data_aug import *
import cv2

def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    cls = int(line_parts[1])
    landmarks = None
    if len(line_parts)>2:
       landmarks = list(map(float, line_parts[2: len(line_parts)]))
    return img_name, cls, landmarks


class FaceLandmarksDataset(Dataset):
    def __init__(self, src_lines, phase, transform=None):
        '''
        :param src_lines: src_lines
        :param train: whether we are training or not
        :param transform: data transform
        '''
        self.lines = src_lines
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        # rect:(left, upper, right, lower)
        img_name, cls, landmarks = parse_line(self.lines[idx])
        if landmarks is not None:
            landmarks = np.array(landmarks).astype(np.float32)
            landmarks = landmarks.reshape(-1, 2)  # 转成x,y格式，便于后面操作
        else:
            landmarks = np.zeros((21,2))
        # image
        img_bgr = cv2.imread(img_name)  # HxWxC
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        #此处保留原图和原始的landmark是为后面的rotate服务，如果直接rotate人脸区域，会产生很多黑色区域
        sample = {'image': img,'class':cls,'landmarks': landmarks,'path':img_name}
        if self.transform is not None:
            sample = self.transform(sample)
            sample['class']=cls
        return sample


def load_data(data_file, transform):
    with open(data_file) as f:
        lines = f.readlines()
    data_set = FaceLandmarksDataset(lines, data_file, transform)
    return data_set


def get_train_test_set():
    tran_tsf = transforms.Compose([
        Normalize(),  # do channel normalization
        ToTensor()]  # convert to torch type: NxCxHxW
    )
    val_tsf = transforms.Compose([
        Normalize(),  # do channel normalization
        ToTensor()]  # convert to torch type: NxCxHxW
    )
    train_set = load_data('stage3_train_list.txt', transform=tran_tsf)
    valid_set = load_data('stage3_val_list.txt', transform=val_tsf)
    print(len(train_set),len(valid_set))
    return train_set, valid_set


if __name__ == '__main__':
    tsfm = transforms.Compose([
        Normalize(),  # do channel normalization
        ToTensor()]  # convert to torch type: NxCxHxW
    )
    train_set = load_data('../stage3_train_list.txt', tsfm)
    for i in range(0, len(train_set)):
        sample = train_set[i]
        path = sample['path']
        img = sample['image'].squeeze(0).numpy()
        h, w = img.shape[:2]
        img = img*np.std(img)+np.mean(img)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        landmarks = sample['landmarks']  # 42
        if landmarks is not None:
            # 请画出人脸crop以及对应的landmarks
            #please complete your code under this blank
            landmarks = landmarks.reshape(-1, 2)*112
            for idx, (x, y) in enumerate(landmarks):
                cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
        cv2.imshow("ori", img)
        key = cv2.waitKey(0)
        if key == 27:
            exit(0)
        cv2.destroyAllWindows()
