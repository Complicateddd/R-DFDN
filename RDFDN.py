from utils import correlation_reg
from resnet_base import ResNet, MLPLayer
import torch
import torch.nn as nn
from FALoss import FALoss, DiffLoss
from endecoder import decoder

class R_DFDN(nn.Module):
    def __init__(self, ):
        super(R_DFDN, self).__init__()
        self.inv_encoder = ResNet(depth=56, nb_filters=16, num_classes=10, bn=False, kernel_size=3, inp_channels=3, k=1,
                               affine=True, inp_noise=0)
        self.diff_encoder = ResNet(depth=56, nb_filters=16, num_classes=10, bn=False, kernel_size=3, inp_channels=3, k=1,
                                  affine=True, inp_noise=0)

        self.tar_de = decoder()

        self.classificator1 = nn.Sequential(
            nn.Linear(100,100),
            nn.ReLU(True),
            nn.Linear(100,10)
        )
        self.classificator2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )

        self.inv_source_diff = nn.Sequential(
            nn.Linear(64*7*7,100),
            nn.ReLU()
        )
        self.inv_tar_diff = nn.Sequential(
            nn.Linear(64 * 7 * 7, 100),
            nn.ReLU()
        )

        self.cha_tar_diff = nn.Sequential(
            nn.Linear(64 * 7 * 7, 100),
            nn.ReLU()
        )



    def forward(self, sourse_input, tar_input, is_generation=True, is_train=True):
        hid, source_inv = self.inv_encoder(sourse_input)
        _, tar_inv = self.inv_encoder(tar_input)
        source_inv_diff = self.inv_source_diff(source_inv.view(source_inv.size(0),-1))
        tar_inv_diff = self.inv_tar_diff(tar_inv.view(tar_inv.size(0),-1))

        _, tar_diff = self.diff_encoder(tar_input)
        tar_cha_diff = self.cha_tar_diff(tar_diff.view(tar_diff.size(0),-1))

        tar_rec = self.tar_de(tar_inv_diff+tar_cha_diff)

        source_class = source_inv_diff
        tar_class = source_inv_diff + tar_cha_diff


        fc = torch.mean(source_class.view(source_class.size(0), source_class.size(1), -1), dim=2)
        fc = fc.view(fc.size()[0], -1)
        source_pred = self.classificator1((fc))


        fc2 = torch.mean(tar_class.view(tar_class.size(0), tar_class.size(1), -1), dim=2)
        fc2 = fc2.view(fc2.size()[0], -1)
        tar_pred = self.classificator2((fc2))

        if is_train:
            return hid, source_pred,tar_pred,source_inv_diff,tar_inv_diff,tar_cha_diff,tar_rec,source_inv,tar_inv
        elif is_generation:
            return source_pred
        else:
            return tar_pred

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.class_loss_criterion = nn.CrossEntropyLoss()
        self.class_loss_criterion2 = nn.CrossEntropyLoss()
        self.recognition_loss_criterion = nn.L1Loss()
        self.regulazation_loss_criterion = correlation_reg
        self.FA = FALoss()
        self.DIFF = DiffLoss()

    def forward(self, hid, source_pred,tar_pred,targets,source_inv_diff,tar_inv_diff,tar_cha_diff,tar_rec,tar_img,source_inv,tar_inv,
                weight_list=[0.1,1,1,0.1,0.1,0.1]):
    
        hid_1_loss = self.regulazation_loss_criterion(hid, targets.cpu(), within_class=True, subtract_mean=True)
        s_class_loss = self.class_loss_criterion(source_pred, targets)
        t_class_loss = self.class_loss_criterion2(tar_pred, targets)
        t_recognition_loss = self.recognition_loss_criterion(tar_rec, tar_img)
        fa_loss = self.FA(source_inv,tar_inv)
        t_diff_loss = self.DIFF(tar_inv_diff,tar_cha_diff)
        totol_loss = hid_1_loss * weight_list[0] + s_class_loss * weight_list[1] +t_class_loss * weight_list[2]\
                      +t_recognition_loss * weight_list[3]+fa_loss * weight_list[4]  +  t_diff_loss * weight_list[5]
        return hid_1_loss, s_class_loss, t_class_loss, t_recognition_loss,fa_loss,t_diff_loss,totol_loss


if __name__ == '__main__':
    pass