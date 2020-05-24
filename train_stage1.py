#encoding=utf-8
from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
from dataset.stage1_dataset import get_train_test_set
from torch.utils.data.sampler import SubsetRandomSampler
from models.stage1_baseline import BaseNet
from models.mobilenet_v1 import MobileNetV1
from models.mbv1_fpn import MobileNetV1FPN
from torch.optim.lr_scheduler import MultiStepLR,ReduceLROnPlateau
import cv2
import numpy as np
import torch.nn.functional as F

torch.set_default_tensor_type(torch.FloatTensor)

parser = argparse.ArgumentParser(description='Detector')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 100)')
parser.add_argument('--network', type=str, default="base",
                        help='network for trainning:base | mbv1 | mbv1fpn')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--resume_weights', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=4e-5, type=float, help='Weight decay for SGD')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
parser.add_argument('--save_model', action='store_true', default=True,
                        help='save the current Model')
parser.add_argument('--save_directory', type=str, default='trained_models/base/stage1',
                        help='learnt models are saving here')
parser.add_argument('--phase', type=str, default='Train',  # Train/train, Predict/predict,Test/test,Finetune/finetune
                        help='training, predicting or finetuning')

def eval(valid_loader,model,device):
    model.to(device)
    model.eval()  # prep model for evaluation
    with torch.no_grad():
        for valid_batch_idx, batch in enumerate(valid_loader):
            valid_img = batch['image']
            landmark = batch['landmarks']
            input_img = valid_img.to(device)
            target_pts = landmark.to(device)
            output_pts = model(input_img)
            for i in range(output_pts.shape[0]):
                landmarks = output_pts[i].reshape(-1, 2) * 112
                t_landmarks = target_pts[i].reshape(-1, 2) * 112
                img = valid_img[i].squeeze(0).numpy()
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                img = img.astype('float32')
                img_r = img.copy()
                for idx, (x, y) in enumerate(landmarks):
                    cv2.circle(img_r, (int(x), int(y)), 1, (0, 255, 0), -1)
                for idx, (x, y) in enumerate(t_landmarks):
                    cv2.circle(img_r, (int(x), int(y)), 1, (0, 0, 255), -1)

                cv2.imshow("face", img_r)
                key = cv2.waitKey(0)

def predict(model,device,img_path):
    model.eval()
    img_bgr = cv2.imread("./image/t3.jpg")
    img_bgr = cv2.resize(img_bgr, (112, 112))
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32')
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / (std + 0.000001)
    img = img[np.newaxis, :, :]
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
    output = model(img_tensor)
    print(output)
    landmarks = output.reshape(-1, 2) * 112
    for idx, (x, y) in enumerate(landmarks):
        cv2.circle(img_bgr, (int(x), int(y)), 1, (0, 255, 0), -1)
        # cv2.putText(img_bgr, str(idx), (int(x), int(y)), 1, 0.5, (0, 0, 255))
    cv2.imshow("face", img_bgr)
    key = cv2.waitKey(0)

def validate(valid_loader,model,device,criterion):
    ######################
    # validate the model #
    ######################
    valid_mean_pts_loss = 0.0

    model.eval()  # prep model for evaluation
    with torch.no_grad():
        for valid_batch_idx, batch in enumerate(valid_loader):
            valid_img = batch['image']
            landmark = batch['landmarks']

            input_img = valid_img.to(device)
            target_pts = landmark.to(device)
            output_pts = model(input_img)
            valid_loss = criterion(target_pts,output_pts)
            valid_mean_pts_loss += valid_loss.item()
        return np.mean(valid_mean_pts_loss)

def train(args, train_loader, valid_loader, model, criterion, optimizer,scheduler, device):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    train_losses = []
    valid_losses = []

    for start_epoch in range(args.resume_epoch,epoch):
        ######################
        # training the model #
        ######################
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']
            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()

            # get output
            output_pts = model(input_img)
            # get loss
            loss = criterion(output_pts, target_pts)
            train_losses.append(loss.item())
            # do BP automatically
            loss.backward()
            optimizer.step()
        scheduler.step()
            # scheduler.step(val_loss)
        if start_epoch %5==0:
            val_pts_loss = validate(valid_loader, model, device, criterion)
            val_loss = val_pts_loss.item()
            valid_losses.append(val_pts_loss)
            print('Train Epoch: {} pts_loss: {:.6f},Valid:pts_loss: {:.6f} LR:{:.5f}'.format(
                start_epoch, loss.item(), val_loss, optimizer.state_dict()['param_groups'][0]['lr']))
        if start_epoch % 100 ==0:
            # save model
            if args.save_model:
                saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(start_epoch) + '.pt')
                torch.save(model.state_dict(), saved_model_name)
    saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_Final.pt')
    torch.save(model.state_dict(), saved_model_name)
    return train_losses, valid_losses


def main():
    args = parser.parse_args()
    ###################################################################################
    torch.manual_seed(args.seed)
    # For single GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")  # cuda:0

    print('===> Loading Datasets')
    train_set, test_set= get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,num_workers=2)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size,num_workers=2)

    print('===> Building Model')
    model =BaseNet()
    model = model.to(device)
    ########################Optimizer############################################
    criterion_pts = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[800, 900], gamma=0.1)

    #scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=120,verbose=True)
    ####################################################################
    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        train_losses, valid_losses = \
        train(args, train_loader, valid_loader, model, criterion_pts, optimizer,scheduler, device)
        print('====================================================')
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        model.load_state_dict(torch.load(os.path.join(args.save_directory, 'detector_epoch_300.pt')))
        model.eval()  # prep model for evaluation
        with torch.no_grad():
            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_img = batch['image']
                landmark = batch['landmarks']
                input_img = valid_img.to(device)
                target_pts = landmark.to(device)
                output_pts = model(input_img)
                for i in range(output_pts.shape[0]):
                    landmarks = output_pts[i].reshape(-1, 2) * 112
                    t_landmarks = target_pts[i].reshape(-1, 2) * 112
                    img = valid_img[i].squeeze(0).numpy()
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                    img = img.astype('float32')
                    img_r = img.copy()
                    for idx, (x, y) in enumerate(landmarks):
                        cv2.circle(img_r, (int(x), int(y)), 1, (0, 255, 0), -1)
                    for idx, (x, y) in enumerate(t_landmarks):
                        cv2.circle(img_r, (int(x), int(y)), 1, (0, 0, 255), -1)


                    cv2.imshow("face", img_r)
                    cv2.waitKey(0)

    # how to do finetune?
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        if args.resume_weights is not None:
            print('Loading resume network...')
            state_dict = torch.load(args.resume_weights)
            model.load_state_dict(state_dict)
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, scheduler, device)

    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        files = os.listdir('./image')
        # how to do predict?
        model.load_state_dict(torch.load(os.path.join(args.save_directory, 'detector_epoch_2900.pt')))
        model.eval()
        for file in files:
            img_bgr = cv2.imread("./image/"+file)
            img_bgr = cv2.resize(img_bgr, (112, 112))
            img = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
            img = img.astype('float32')
            mean = np.mean(img)
            std = np.std(img)
            img=(img-mean)/(std+0.000001)
            img = img[np.newaxis,:,:]
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).to(device)
            output = model(img_tensor)
            print(output)
            landmarks = output.reshape(-1, 2)*112
            for idx, (x, y) in enumerate(landmarks):
                cv2.circle(img_bgr, (int(x), int(y)), 1, (0, 255, 0), -1)
                #cv2.putText(img_bgr, str(idx), (int(x), int(y)), 1, 0.5, (0, 0, 255))
            cv2.imshow("face", img_bgr)
            cv2.waitKey(0)

if __name__ == '__main__':
    main()

