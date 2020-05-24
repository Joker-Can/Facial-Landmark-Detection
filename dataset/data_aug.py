import numpy as np
import cv2
import torch
import random
train_boarder = 112

'''
lesson1:rotation+translation
M=[[cosa -sina tx],[sina cosa ty]]
(x,y)=M*(x0,y0,1).T
https://blog.csdn.net/LZH2912/article/details/78712881
'''
def rotate(image, landmarks):
    imgT = None
    while 1:
        angle = np.random.randint(-30, 30)
        h, w = image.shape[:2]
        landmarks = landmarks.reshape(-1, 2)
        x1, y1, x2, y2, center, bbox_size = expand_roi(w, h, landmarks)
        M = cv2.getRotationMatrix2D((center[0][0], center[0][1]), angle, 1)
        # 旋转原图
        imgT = cv2.warpAffine(image, M, (int(h * 1.1), int(w * 1.1)))
        # 旋转原图上的landmarks
        landmarksT = np.asarray([(M[0, 0] * x + M[0, 1] * y + M[0, 2],
                                  M[1, 0] * x + M[1, 1] * y + M[1, 2]) for (x, y) in landmarks])
        wh = np.ptp(landmarksT, axis=0).astype(np.int32) + 1
        # np.ceil向上取整, 运算称为Ceiling，扩展1.25倍，然后随机选取
        size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
        # 计算新的左上角坐标
        xy = np.asarray((center[0][0] - size // 2, center[0][1] - size // 2), dtype=np.int32)
        # 归一化坐标点
        landmarksT = landmarksT - xy
        # 检查比例情况，因为这里有随机过程，如果比例不对就重新算
        if (landmarksT / size < 0).any() or (landmarksT / size > 1).any():
            continue
        else:
            x1, y1 = xy
            x2, y2 = xy + size
            height, width = imgT.shape[:2]
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            # 将人脸图片从旋转之后的图片上扣下来
            imgT = imgT[y1:y2, x1:x2]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            image_resize = cv2.resize(imgT, (train_boarder, train_boarder))
            ori_h, ori_w = imgT.shape[:2]
            landmarksT[:, 0] *= 1.0 * train_boarder / ori_w  # 将x坐标缩放到resize之后的尺寸
            landmarksT[:, 1] *= 1.0 * train_boarder / ori_h  # 将y坐标缩放到resize之后的尺寸
            return image_resize, landmarksT


def mirror(image,landms):
    h, w = image.shape[:2]
    landms = landms.reshape(-1, 2)
    img_t = image[:, ::-1].copy()
    landms_t = [(w - x, y) for x, y in landms]  # 计算水平翻转后的坐标
    # 图像水平翻转
    return img_t, np.array(landms_t)
    return image,landms

'''photo-metric colour distortion'''
def distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp
    if image.ndim < 3:
        return image
    image = image.copy()
    if random.randrange(2):
        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))
        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    else:
        #brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        #hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        #contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image

def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean)/(std+0.00000001)
    return pixels

def expand_roi(img_width,img_height,landmarks):   # usually ratio = 0.25
    xy_min = np.min(landmarks, axis=0, keepdims=True).astype(np.int32)
    # 求原始坐标的最大值（原始最右上角）
    xy_max = np.max(landmarks, axis=0, keepdims=True).astype(np.int32)
    # 求landmark的宽度和高度
    wh = xy_max - xy_min + 1
    # 求landmark的中心点(center_x,center_y)
    center = (xy_min + wh / 2).astype(np.int32)
    # 扩大ROI 0.2倍(以最长的边 * 1.2作为boxsize)
    boxsize = int(np.max(wh) * 1.2)
    # 求取扩展后的最左上点
    tl = center - boxsize // 2
    # 扩展后最左上点
    x1, y1 = tl[0, 0], tl[0, 1]
    # 扩展后最右下点
    br = tl + boxsize
    x2, y2 = br[0, 0], br[0, 1]
    # 避免坐标出现负值
    x1 = max(0, x1)
    y1 = max(0, y1)
    # 限制bbox不超过原图
    x2 = min(img_width, x2)
    y2 = min(img_height, y2)
    return x1,y1,x2,y2,center,boxsize

class Resize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_resize = cv2.resize(image,(train_boarder, train_boarder))
        ori_h,ori_w=image.shape[:2]
        landmarks[:, 0]*=1.0*train_boarder/ori_w  #将x坐标缩放到resize之后的尺寸
        landmarks[:, 1] *= 1.0 * train_boarder / ori_h #将y坐标缩放到resize之后的尺寸
        ori_landmark = None
        if 'ori_landmark' in sample.keys():
            ori_landmark = sample['ori_landmark']
        return {'image': image_resize,'landmarks': landmarks,'path':sample['path'],'ori_landmark':ori_landmark}

class Normalize(object):

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        back = image.copy()
        image = channel_norm(image)
        if landmarks is not None:
            landmarks /= train_boarder  # 归一化
        ori_landmark = None
        if 'ori_landmark'  in sample.keys():
            ori_landmark = sample['ori_landmark']
        return {'image': image,'landmarks': landmarks,'path':sample['path'],'ori_landmark':ori_landmark}

class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        cls = torch.tensor(0).float()
        if 'class' in sample.keys():
            cls=sample['class']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image[np.newaxis,:,:]
        landmarks_tensor = None
        if landmarks is not None:
            landmarks = landmarks.reshape(-1,)
            landmarks_tensor = torch.from_numpy(landmarks).float()
        sample_ =  {'image': torch.from_numpy(image).float(),
                'landmarks': landmarks_tensor,'path':sample['path']}
        if 'class' in sample.keys():
            sample_['class'] = cls
        return sample_

class Distort(object):

    def __init__(self,prob):
        self.prob = prob

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        if random.random() < self.prob:
            image = distort(image)  # colour distortion
        ori_landmark = None
        if 'ori_landmark' in sample.keys():
            ori_landmark = sample['ori_landmark']
        return {'image': image, 'landmarks': landmarks,'path':sample['path'],'ori_landmark':ori_landmark}

class Rotate(object):

    def __init__(self,prob):
        self.prob = prob

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        ori_landmark = None
        if 'ori_landmark' in sample.keys():
            ori_landmark = sample['ori_landmark']
        else:
            print("ori_landmark empty")
            return {'image': image, 'landmarks': landmarks, 'path': sample['path'], 'ori_landmark': ori_landmark}
        if random.random() < self.prob:
            ori_image = cv2.imread(sample['path']) #原图
            ori_lamdmarks = sample['ori_landmark']
            image, landmarks = rotate(ori_image, ori_lamdmarks)
        return {'image': image, 'landmarks': landmarks,'path':sample['path'],'ori_landmark':ori_landmark}

class Mirror(object):

    def __init__(self,prob):
        self.prob = prob

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        if random.random() < self.prob:
            image, landmarks = mirror(image, landmarks)
        ori_landmark = None
        if 'ori_landmark' in sample.keys():
            ori_landmark = sample['ori_landmark']
        return {'image': image, 'landmarks': landmarks,'path':sample['path'],'ori_landmark':ori_landmark}