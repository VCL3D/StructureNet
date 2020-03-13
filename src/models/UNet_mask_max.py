import torch
import torch.nn as nn
import math
from .blocks import *


class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        """
        Args:
            size: expected size after interpolation
            mode: interpolation type (e.g. bilinear, nearest)
        """
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        out = self.interp(x, size=self.size, mode=self.mode) #, align_corners=False
        
        return out


class Encoder(nn.Module):
    def __init__(self, ndf):
        super(Encoder, self).__init__()

        self.ndf = ndf

        self.encoder_conv1_1 = get_conv_relu(1, self.ndf, kernel_size=7, stride=1, padding=3) #PartialConv2dBlock(1, ndf, bn=False, activ='elu', sample='none-7') 
        self.encoder_conv1_2 = get_conv_relu(self.ndf, self.ndf * 2, kernel_size=5, stride=1, padding=2) #PartialConv2dBlock(ndf, ndf * 2, bn=False, activ='elu', sample='none-5') 
        
        self.encoder_pool_1 = get_max_pool(kernel_size=2, stride=2)

        self.encoder_conv2_1 = get_conv_relu(self.ndf * 2, self.ndf * 4, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 2, ndf * 4, bn=False, activ='elu', sample='down-3') 
        self.encoder_conv2_2 = get_conv_relu(self.ndf * 4, self.ndf * 4, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 4, ndf * 4, bn=False, activ='elu') 
        self.encoder_conv2_3 = get_conv_relu(self.ndf * 4, self.ndf * 4, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 4, ndf * 4, bn=False, activ='elu') 
        
        self.encoder_pool_2 = get_max_pool(kernel_size=2, stride=2)

        self.encoder_conv3_1 = get_conv_relu(self.ndf * 4, self.ndf * 8, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 4, ndf * 8, bn=False, activ='elu', sample='down-3')
        self.encoder_conv3_2 = get_conv_relu(self.ndf * 8, self.ndf * 8, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu') 
        self.encoder_conv3_3 = get_conv_relu(self.ndf * 8, self.ndf * 8, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu')
        
        self.encoder_pool_3 = get_max_pool(kernel_size=2, stride=2)

        self.encoder_conv4_1 = get_conv_relu(self.ndf * 8, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        self.encoder_conv4_2 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        self.encoder_conv4_3 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        
        self.encoder_pool_4 = get_max_pool(kernel_size=2, stride=2)

        self.encoder_conv5_1 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        self.encoder_conv5_2 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        self.encoder_conv5_3 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.encoder_conv1_1(x)
        out_pre_ds_1 = self.encoder_conv1_2(out)
        out = self.encoder_pool_1(out_pre_ds_1)
        out = self.encoder_conv2_1(out)
        out = self.encoder_conv2_2(out)
        out_pre_ds_2 = self.encoder_conv2_3(out)
        out = self.encoder_pool_2(out_pre_ds_2)
        out = self.encoder_conv3_1(out)
        out = self.encoder_conv3_2(out)
        out_pre_ds_3 = self.encoder_conv3_3(out)
        out = self.encoder_pool_3(out_pre_ds_3)
        out = self.encoder_conv4_1(out)
        out = self.encoder_conv4_2(out)
        out_pre_ds_4 = self.encoder_conv4_3(out)
        out = self.encoder_pool_4(out_pre_ds_4)
        out = self.encoder_conv5_1(out)
        out = self.encoder_conv5_2(out)
        out = self.encoder_conv5_3(out)

        return out_pre_ds_1, out_pre_ds_2, out_pre_ds_3, out_pre_ds_4, out


class Latent(nn.Module):
    def __init__(self, ndf):
        super(Latent, self).__init__()

        self.ndf = ndf

        self.encoder_resblock1 = ResConv(self.ndf * 16) 
        self.encoder_resblock2 = ResConv(self.ndf * 16)
        self.encoder_resblock3 = ResConv(self.ndf * 16)
        self.encoder_resblock3 = ResConv(self.ndf * 16)
        self.conv_1x1 = get_conv(self.ndf * 16, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.encoder_resblock1(x)
        out = self.encoder_resblock2(out)
        out = self.encoder_resblock3(out)
        attention = self.conv_1x1(out)
        attention = nn.functional.sigmoid(attention)
        out = attention*out

        return out, attention


class Decoder(nn.Module):
    def __init__(self, width, height, ndf, upsample, nclasses):
        super(Decoder, self).__init__()

        self.h = height
        self.w = width
        self.ndf = ndf
        self.upsample = upsample
        self.nclasses = nclasses

        #self.decoder_upsample4 = Interpolate((22, self.w // 8), mode=self.upsample)
        self.decoder_upsample4 = Interpolate((self.h // 8, self.w // 8), mode=self.upsample)
        self.decoder_deconv5_3 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        self.decoder_deconv5_2 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        self.decoder_deconv5_1 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        self.decoder_conv_id_4 = get_conv_relu(2 * self.ndf * 16, self.ndf * 16, kernel_size=1, stride=1, padding=0)
        self.decoder_deconv4_3 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        self.decoder_deconv4_2 = get_conv_relu(self.ndf * 16, self.ndf * 16, kernel_size=3, stride=1, padding=1)
        self.decoder_upsample3 = Interpolate((self.h // 4, self.w // 4), mode=self.upsample)
        self.decoder_deconv4_1 = get_conv_relu(self.ndf * 16, self.ndf * 8, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu') 
        self.decoder_conv_id_3 = get_conv_relu(2 * self.ndf * 8, self.ndf * 8, kernel_size=1, stride=1, padding=0) #conv_1x1(2 * ndf * 8, ndf * 8, n_type=self.type)
        self.decoder_deconv3_3 = get_conv_relu(self.ndf * 8, self.ndf * 8, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu') 
        self.decoder_deconv3_2 = get_conv_relu(self.ndf * 8, self.ndf * 8, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 8, ndf * 8, bn=False, activ='elu') 
        self.decoder_upsample2 = Interpolate((self.h // 2, self.w // 2), mode=self.upsample)
        self.decoder_deconv3_1 = get_conv_relu(self.ndf * 8, self.ndf * 4, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 8, ndf * 4, bn=False, activ='elu') 
        self.decoder_conv_id_2 = get_conv_relu(2 * self.ndf * 4, self.ndf * 4, kernel_size=1, stride=1, padding=0) #conv_1x1(2 * ndf * 4, ndf * 4, n_type=self.type)
        self.decoder_deconv2_3 = get_conv_relu(self.ndf * 4, self.ndf * 4, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 4, ndf * 4, bn=False, activ='elu') 
        self.decoder_deconv2_2 = get_conv_relu(self.ndf * 4, self.ndf * 4, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 4, ndf * 4, bn=False, activ='elu') 
        self.decoder_upsample1 = Interpolate((self.h, self.w), mode=self.upsample)
        self.decoder_deconv2_1 = get_conv_relu(self.ndf * 4, self.ndf * 2, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 4, ndf * 2, bn=False, activ='elu') 
        self.decoder_conv_id_1 = get_conv_relu(2 * self.ndf * 2, self.ndf * 2, kernel_size=1, stride=1, padding=0) #conv_1x1(2 * ndf * 2, ndf * 2, n_type=self.type)
        self.decoder_deconv1_2 = get_conv_relu(self.ndf * 2, self.ndf, kernel_size=3, stride=1, padding=1) #PartialConv2dBlock(ndf * 2, ndf, bn=False, activ='elu')
        self.decoder_deconv1_1 = get_conv(self.ndf, self.nclasses, kernel_size=1, stride=1, padding=0) #PartialConv2dBlock(ndf, 1, bn=False, activ='no_acitv')

    def forward(self, x):
        out = self.decoder_deconv5_3(x[4])
        out = self.decoder_deconv5_2(out)
        out = self.decoder_upsample4(out)
        mask4 = self.decoder_upsample4(x[5])
        out = self.decoder_deconv5_1(out)
        out_post_up_4 = torch.cat((out, x[3]), 1)
        out = self.decoder_conv_id_4(out_post_up_4)
        out = out*mask4
        out = self.decoder_deconv4_3(out)
        out = self.decoder_deconv4_2(out)
        out = self.decoder_upsample3(out)
        mask3 = self.decoder_upsample3(x[5])
        out = self.decoder_deconv4_1(out)
        out_post_up_3 = torch.cat((out, x[2]), 1)
        out = self.decoder_conv_id_3(out_post_up_3)
        out = out*mask3
        out = self.decoder_deconv3_3(out)
        out = self.decoder_deconv3_2(out)
        out = self.decoder_upsample2(out)
        mask2 = self.decoder_upsample2(x[5])
        out = self.decoder_deconv3_1(out)
        out_post_up_2 = torch.cat((out, x[1]), 1)
        out = self.decoder_conv_id_2(out_post_up_2)
        out = out*mask2
        out = self.decoder_deconv2_3(out)
        out = self.decoder_deconv2_2(out)
        out = self.decoder_upsample1(out)
        mask1 = self.decoder_upsample1(x[5])
        out = self.decoder_deconv2_1(out)
        out_post_up_1 = torch.cat((out, x[0]), 1)
        out = self.decoder_conv_id_1(out_post_up_1)
        out = out*mask1
        out_for_vis = self.decoder_deconv1_2(out)
        out = self.decoder_deconv1_1(out_for_vis)
         #pred = nn.functional.log_softmax(out, dim=1)#changed only to be compatible with ONNX exporter
        pred = nn.functional.softmax(out,dim = 1).log()

        return out_for_vis, pred


class UNet_mask_max(nn.Module):
    def __init__(self, width, height, ndf, upsample, nclasses):
        super(UNet_mask_max, self).__init__()
        """
        Args:
            width: input width
            height: input height
            ndf: constant number from channels
            upsample: upsampling type (nearest | bilateral)
            nclasses: number of semantice segmentation classes
        """
        self.h = height
        self.w = width
        self.ndf = ndf
        self.upsample = upsample
        self.nclasses = nclasses

        self.encoder = Encoder(self.ndf)
        self.latent = Latent(self.ndf)
        self.decoder = Decoder(self.w, self.h, ndf, self.upsample, self.nclasses)

    def forward(self, x):
        out_list = []
        out_pre_ds_1, out_pre_ds_2, out_pre_ds_3, out_pre_ds_4, out = self.encoder(x)

        # out_list.append(out_pre_ds_1)
        # out_list.append(out_pre_ds_2)
        # out_list.append(out_pre_ds_3)
        # out_list.append(out_pre_ds_4)

        out, attention = self.latent(out)
        # out_list.append(out)
        # out_list.append(attention)

        out_list = (out_pre_ds_1,out_pre_ds_2,out_pre_ds_3,out_pre_ds_4,out,attention)
        
        activs, seg_out = self.decoder(out_list)

        return activs, seg_out