# encoding=utf-8
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.data_aug import *


def parse_line(line):
    line_parts = line.strip().split()
    img_name = line_parts[0]
    rect = list(map(int, list(map(float, line_parts[1:5]))))
    landmarks = list(map(float, line_parts[5: len(line_parts)]))
    return img_name, rect, landmarks


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
        img_name, rect, landmarks = parse_line(self.lines[idx])
        landmarks = np.array(landmarks).astype(np.float32)
        ori_landmarks = landmarks.copy()
        landmarks = landmarks.reshape(-1, 2)  # 转成x,y格式，便于后面操作
        # image
        img_bgr = cv2.imread(img_name)  # HxWxC
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_height, img_width = img.shape
        x1,y1,x2,y2,_,_ = expand_roi(img_width, img_height, landmarks) #得到在原图上扩充后的bbox
        # 不会超出原图（截取bbox的图像数据）
        imgT = img[y1:y2, x1:x2]
        landmarks[:, 0] -= x1  # 将x坐标与crop之后的image对齐
        landmarks[:, 1] -= y1  # 将y坐标与crop之后的image对齐
        #此处保留原图和原始的landmark是为后面的rotate服务，如果直接rotate人脸区域，会产生很多黑色区域
        sample = {'image': imgT, 'landmarks': landmarks,'path':img_name}
        if self.transform is not None:
            sample['ori_landmark'] = ori_landmarks
            sample = self.transform(sample)
            imgT = sample['image'].squeeze(0).numpy()
            if len(imgT.shape)==3:
                imgT_gray = cv2.cvtColor(imgT, cv2.COLOR_BGR2GRAY)
                sample['image']=torch.from_numpy(imgT_gray).float().unsqueeze(0)
        return sample


def load_data(data_file, transform):
    with open(data_file) as f:
        lines = f.readlines()
    data_set = FaceLandmarksDataset(lines, data_file, transform)
    return data_set


def get_train_test_set():
    tran_tsf = transforms.Compose([
        Resize(),
        Rotate(prob=0.3),
        #Distort(prob=0.2),
        #Mirror(prob=0.2),
        Normalize(),  # do channel normalization
        ToTensor()]  # convert to torch type: NxCxHxW
    )
    val_tsf = transforms.Compose([
        Resize(),
        Normalize(),  # do channel normalization
        ToTensor()]  # convert to torch type: NxCxHxW
    )
    train_set = load_data('train_list.txt', transform=tran_tsf)
    valid_set = load_data('val_list.txt', transform=val_tsf)
    return train_set, valid_set


if __name__ == '__main__':
    tsfm = transforms.Compose([
        Resize(),
        Rotate(prob=0.3),
        # Distort(prob=0.2),
        #Mirror(prob=0.2),
        Normalize(),  # do channel normalization
        ToTensor()]  # convert to torch type: NxCxHxW
    )
    train_set = load_data('../stage1_train_list.txt', tsfm)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=5, shuffle=True)

    for i in range(0, len(train_set)):
        sample = train_set[i]
        path = sample['path']

        img = sample['image'].squeeze(0).numpy()

        h, w = img.shape[:2]

        img = img*np.std(img)+np.mean(img)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        landmarks = sample['landmarks']  # 42
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
