import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import torch.optim as optim

import glob
import os
import os.path


import os
from PIL import Image
import argparse


class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = os.listdir(root)
        self.class_to_idx = {c: int(c) for i, c in enumerate(self.classes)}
        self.imgs = []
        for c in self.classes:
            class_dir = os.path.join(root, c)
            for filename in os.listdir(class_dir):
                path = os.path.join(class_dir, filename)
                self.imgs.append((path, self.class_to_idx[c])) 
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, target




####################
#If you want to use your own custom model
#Write your code here
####################
# class Custom_model(nn.Module):
#     def __init__(self):
#         super(Custom_model, self).__init__()
#         #place your layers
#         #CNN, MLP and etc.
#
#     def forward(self, input):
#         #place for your model
#         #Input: 3* Width * Height
#         #Output: Probability of 50 class label
#         return predicted_label



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

####################
#Modify your code here
####################
def model_selection(selection):
    if selection == "resnet":
        model = models.resnet18(weights='DEFAULT')
        model.conv1 =  nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False)
        model.layer4 = Identity()
        model.fc = nn.Linear(256, 50)
    elif selection == "vgg":
        model = models.vgg11_bn(weights='DEFAULT')
        model.features = nn.Sequential(*list(model.features.children())[:-7])
        model.classifier = nn.Sequential( nn.Linear(in_features=25088, out_features=50, bias=True))
    elif selection == "mobilenet":
        model = models.mobilenet_v2(weights='DEFAULT')
        model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=50, bias=True))
    # elif  selection =='custom':
    #     model = Custom_model()
    return model



# def train(net1, labeled_loader, optimizer, criterion, scheduler):
#
#     net1.train()
#     #Supervised_training
#     for batch_idx, (inputs, targets) in enumerate(labeled_loader):
#         if torch.cuda.is_available():
#             inputs, targets = inputs.cuda(), targets.cuda()
#         optimizer.zero_grad()
#         ####################
#         #Write your Code
#         #Model should be optimized based on given "targets"
#         ####################
#         outputs = net1(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#     scheduler.step()
#     # print("have been scheduler.step()")
        
def train(net1, labeled_loader, optimizer, criterion):
    net1.train()
    for batch_idx, (inputs, targets) in enumerate(labeled_loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net1(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


        
# def test(net, testloader):
#     net.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             if torch.cuda.is_available():
#                 inputs, targets = inputs.cuda(), targets.cuda()
#             outputs = net(inputs)
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#         return 100. * correct / total

def test(net, testloader, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return 100. * correct / total, test_loss / len(testloader)






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--test',  type=str,  default='False')
    parser.add_argument('--student_abs_path',  type=str,  default='./')
    args = parser.parse_args()



    if not os.path.exists(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning')):
        os.makedirs(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning'))



    batch_size = 256      #Input the number of batch size
    if args.test == 'False':
        train_transform = transforms.Compose([
                    transforms.RandomResizedCrop(64, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        test_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        
        dataset = CustomDataset(root = './data/Supervised_Learning/labeled', transform = train_transform)
        labeled_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        dataset = CustomDataset(root = './data/Supervised_Learning/val', transform = test_transform)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    else :
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    """반복 실행을 위해서 추가한 부분"""
    model_list = ['mobilenet']
    step_size = [4,3,2]
    factor = [0.1,0.075,0.05]


    for model_for in range(len(model_list)):
        total_log = open('./Test_log/{}_re_total_Log.txt'.format(model_list[model_for]),'w')

        for step_for in range(len(step_size)):
            for factor_for in range(len(factor)):

                model_name = model_list[model_for]
                print("Model = {}".format(model_name))
                # Input model name to use in the model_section class
                # e.g., 'resnet', 'vgg', 'mobilenet', 'custom'

                if torch.cuda.is_available():
                    model = model_selection(model_name).cuda()
                    print("CUDA Available!")
                else:
                    model = model_selection(model_name)

                params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

                # You may want to write a loader code that loads the model state to continue the learning process
                # Since this learning process may take a while.

                if torch.cuda.is_available():
                    criterion = nn.CrossEntropyLoss().cuda()
                else:
                    criterion = nn.CrossEntropyLoss()

                epoch = 45  # Number of Epochs
                optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer with learning rate
                # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)  # LR Scheduler
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=step_size[step_for], factor=factor[factor_for])

                # You may want to add a scheduler for your loss

                best_result = 0
                if args.test == 'False':
                    assert params < 7.0, "Exceed the limit on the number of model parameters"
                    os.mkdir('./Test_log/{}_{}_{}'.format(model_list[model_for],step_size[step_for],factor[factor_for]))
                    f_log = open('./Test_log/{}_{}_{}/Log.txt'.format(model_list[model_for],step_size[step_for],factor[factor_for]),'w')
                    for e in range(0, epoch):
                        # train(model, labeled_loader, optimizer, criterion, scheduler)
                        train_loss = train(model, labeled_loader, optimizer, criterion)
                        tmp_res, val_loss = test(model, val_loader, criterion)  # Assume this function returns validation loss
                        # You can change the saving strategy, but you can't change the file name/path
                        # If there's any difference to the file name/path, it will not be evaluated.
                        print('{}th performance, Accuracy : {}, Learning_rate = {}'.format(e + 1, tmp_res,
                                                                                           optimizer.param_groups[0][
                                                                                               'lr']))
                        f_log.write('\n{}th performance, Accuracy : {}, Learning_rate = {}'.format(e + 1, tmp_res,
                                                                                           optimizer.param_groups[0][
                                                                                               'lr']))
                        scheduler.step(val_loss)  # Here we pass validation loss to the scheduler

                        if best_result < tmp_res:
                            best_result = tmp_res
                            torch.save(model.state_dict(),
                                       os.path.join('./logs', 'Supervised_Learning', 'best_model.pt'))
                            torch.save(model.state_dict(),
                                       os.path.join('./Test_log', '{}_{}_{}'.format(model_list[model_for],step_size[step_for],factor[factor_for]), 'best_model.pt'))
                    print('Final performance {} - {}'.format(best_result, params))
                    f_log.write('\n\nFinal performance {} - {}'.format(best_result, params))
                    f_log.close()
                    r_log = open('./Test_log/{}_{}_{}/{}%.txt'.format(model_list[model_for],step_size[step_for],factor[factor_for],best_result),'w')
                    r_log.close()
                    total_log.write('{} / step_size = {} / factor = {} / RESULT = {}\n'.format(model_list[model_for],step_size[step_for],factor[factor_for],best_result))


                else:
                    # This part is used to evaluate.
                    # Do not edit this part!
                    dataset = CustomDataset(root='/data/23_1_ML_challenge/Supervised_Learning/test',
                                            transform=test_transform)
                    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2,
                                             pin_memory=True)

                    model.load_state_dict(
                        torch.load(os.path.join(args.student_abs_path, 'logs', 'Supervised_Learning', 'best_model.pt'),
                                   map_location=torch.device('cuda')))
                    res = test(model, test_loader)
                    print(res, ' - ', params)

        total_log.close()