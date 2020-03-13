import argparse
import os
import sys
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn

import src.models as models
import data
import src
import src.supervision
import src.supervision.metrics as metrics

import src.dataset.box_pose_dataset_factory as dataset_factory
import src.dataset.depthmap_val_dataset as depthmap_val_dataset
import src.dataset.real_dataloader as real_dataloader
import src.dataset as dataset
import src.io as io
import src.utils as utils
import src.dataset.samplers.pose.pose_sampler as pose_sampler
import numpy as np
import torch
import src.dataset.samplers.intrinsics_generator as intrinsics_generator
import src.other as other
from src.dataset.rendering.box_renderer import BoxRenderFlags
import random
import time

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(args):
    usage_text = (
        "StructureNet train/test executor."
        "Usage:  python main.py [options],"
        "   with [options]:"
    )
    parser = argparse.ArgumentParser(description=usage_text)
    # durations
    parser.add_argument('-e','--epochs', type = int, default=20, help = "Train for a total number of <epochs> epochs.")
    parser.add_argument('-b','--batch_size', type = int, default=5, help = "Train with a <batch_size> number of samples each train iteration.")
    parser.add_argument('--test_batch_size', default=1, type = int, help = "Test with a <batch_size> number of samples each test iteration.")    
    parser.add_argument('-c','--checkpoint_iters', type=int, default=1000, help='Save checkpoint (i.e. weights & optimizer) every <checkpoint_iters> iterations.')
    parser.add_argument('-t','--test_iters', type=int, default=1000, help='Test model every <test_iters> iterations.')
    parser.add_argument('--train_duration', type=int, default = sys.maxsize, help='Train duration counted in iteration.')
    # paths
    parser.add_argument('--train_path', type = str, help = "Path to the training folder containing the files")
    parser.add_argument('--test_path', type = str, help = "Path to the testing folder containing the files")
    weight_group = parser.add_mutually_exclusive_group()
    weight_group.add_argument('--weights', type = str, help = "Path to weights file (for fine-tuning or continuing training)")
    parser.add_argument('--opt_state', type = str, help = "Path to stored optimizer state file (for continuing training)")
    # data paths
    parser.add_argument('--corbs_path', type=str, help = "Path to CORBS background dataset")
    parser.add_argument('--vcl_path', type = str, help = "Path to VCL background dataset")
    parser.add_argument('--intnet_path', type = str, help = 'Path to interior net background dataset')
    parser.add_argument('--valset_path', type = str, help = 'Path to validation set')
    parser.add_argument('--real_data_path', type=str, help= 'Path to real-data set')
    parser.add_argument('--test_data_path', type=str, help= 'Path to rendered data for test')
    #parser.add_argument("--device_list",nargs="*", type=str, default = ["M72e","M72h","M72i","M72j","M11"], help = "List of device names to be loaded")    
    parser.add_argument("--device_list",nargs="*", type=str, default = ["M11"], help = "List of device names to be loaded")    
    #model
    parser.add_argument('-hl', '--heat_weight', type = float, default=0.0, help='Weight/contribution of heatmap loss ot the total loss')
    parser.add_argument('-sl', '--seg_weight', type = float, default=1.0, help='Weight/contribution of segmentation loss ot the total loss')
    parser.add_argument('-cl', '--cor_weight', type = float, default=0.0, help='Weight/contribution of soft correspondences loss')
    parser.add_argument('--soft_cor', type = str2bool, default=False, help='Flag soft correspondences loss using SVD')
    parser.add_argument('-snl', '--surface_weight', type = float, default=0.0, help='Weight/contribution of surface normals loss ot the total loss')
    parser.add_argument('-nc','--nclasses', type = int, default=25, help = "Number of classes.")
    parser.add_argument('--model_name', default="default", type=str, help='Model selection argument.')
    parser.add_argument('--saved_params_path', type=str, help='Path where a trained model has been stored')
    parser.add_argument('--ndf', type=int, default=8, help='Constant values used to define input and output channels at nn layers')
    parser.add_argument('--upsample_type', default="nearest", type=str, help='Model selection argument.')
    # optimization
    parser.add_argument('-o','--optimizer', type=str, default="adam", help='The optimizer that will be used during training.')
    parser.add_argument('-l','--lr', type=float, default=0.0002, help='Optimization Learning Rate.')
    parser.add_argument('-m','--momentum', type=float, default=0.9, help='Optimization Momentum.')
    parser.add_argument('--momentum2', type=float, default=0.999, help='Optimization Second Momentum (optional, only used by some optimizers).')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Optimization Epsilon (optional, only used by some optimizers).')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimization Weight Decay.')
    weight_group.add_argument('--weight_init', type=str, default="xavier", help='Weight initialization method.')
    # hardware
    parser.add_argument('-g','--gpu', type=str, default='0', help='The ids of the GPU(s) that will be utilized. (e.g. 0 or 0,1, or 0,2). Use -1 for CPU.')
    parser.add_argument('--num_workers' ,type= int, default = 3, help='Number of workers to use for dataload')
    # other
    parser.add_argument('-n','--name', type=str, default='default_name', help='The name of this train/test. Used when storing information.')
    parser.add_argument("--depth_thres", type=float, default=5.0, help = "Depth threshold - depth clipping.")
    parser.add_argument("--train_data_type", type=str, help = "Setting enabling type of training data (real, synthetic)", choices=["real", "synthetic", "both"] , default="synthetic")
    
    #dataset parameters
    parser.add_argument("--rmin", type=float, default=1.5, help = "Synthetic dataset params, for more info check PoseSamplerParams.")
    parser.add_argument("--rmax", type=float, default=2.5, help = "Synthetic dataset params, for more info check PoseSamplerParams.")
    parser.add_argument("--zmin", type=float, default=-0.35, help = "Synthetic dataset params, for more info check PoseSamplerParams.")
    parser.add_argument("--zmax", type=float, default=1, help = "Synthetic dataset params, for more info check PoseSamplerParams.")
    parser.add_argument("--lookat", type=float, default=0.5, help = "Synthetic dataset params, for more info check PoseSamplerParams.")
    parser.add_argument("--upvecvar", type=float, default=10.0, help = "Synthetic dataset params, for more info check PoseSamplerParams.")
    parser.add_argument("--samples_per_dataset", type=int, default=50000, help = "Number of samples per dataset.")
    parser.add_argument("--dr", type=float, default=None, help = "Synthetic dataset params, for more info check PoseSamplerParams.")
    parser.add_argument("--dz", type=float, default=None, help = "Synthetic dataset params, for more info check PoseSamplerParams.")
    parser.add_argument("--dphi", type=float, default=None, help = "Synthetic dataset params, for more info check PoseSamplerParams.")
    #visualization
    parser.add_argument('-d','--disp_iters', type=int, default=10, help='Log training progress (i.e. loss etc.) on console every <disp_iters> iterations.')
    parser.add_argument('--visdom', type=str, nargs='?', default=None, const="195.251.117.98", help = "Visdom server IP (port defaults to 8097)")
    parser.add_argument('--visdom_iters', type=int, default=10, help = "Iteration interval that results will be reported at the visdom server for visualization.")
    #validation
    parser.add_argument('--confidence_threshold', type = float, default = 0.75, help ='confidence probability threshold to reject uncofident predictions')

    return parser.parse_known_args(args)

if __name__ == "__main__":
    args, uknown = parse_arguments(sys.argv)
    #create and init device
    print("{} | Torch Version: {}".format(datetime.datetime.now(), torch.__version__))    
    gpus = [int(id) for id in args.gpu.split(',') if int(id) >= 0]
    device = torch.device("cuda:{}" .format(gpus[0]) if torch.cuda.is_available() and len(gpus) > 0 and gpus[0] >= 0 else "cpu")
    print("Training {0} for {1} epochs using a batch size of {2} on {3}".format(args.name, args.epochs, args.batch_size, device))    

    visualizer =src.utils.visualization. NullVisualizer() if args.visdom is None\
        else src.utils.visualization.VisdomVisualizer(args.name, args.visdom, count=4)
    if args.visdom is None:
        args.visdom_iters = 0

    #create model parameters
    model_params = {
        'width': 320,
        'height': 180,
        'ndf': args.ndf,
        'upsample_type': args.upsample_type,
        'nclasses': args.nclasses
    }

    #create & init model || load pretrained model
    if args.saved_params_path is None:
        model = models.get_UNet_model(args.model_name, model_params).to(device)
        other.initialize_weights(model, args.weights if args.weights is not None else args.weight_init)
        model_name = args.model_name
        start_epoch = 0
        iterations = 0
    else:
        checkpoint = torch.load(args.saved_params_path)
        model = models.get_UNet_model(checkpoint['model_name'], model_params)
        print("Loading previously saved model from {}".format(args.saved_params_path))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model_name = checkpoint['model_name']
        start_epoch = checkpoint['epoch']
        iterations = checkpoint['iterations']
    
    #create and init optimizer
    opt_params = other.OptimizerParameters(learning_rate=args.lr, momentum=args.momentum, \
        momentum2=args.momentum2, epsilon=args.epsilon)
    optimizer = other.get_optimizer(args.optimizer, model.parameters(), opt_params)
    if args.saved_params_path is not None:
        print("Loading previously saved optimizer state from {}".format(args.saved_params_path))
        optimizer.load_state_dict(checkpoint["optim_dict"])
    
    #create data importer
    sample_count_per_subdataset = args.samples_per_dataset
    
    path_to_corbs_background_dataset = args.corbs_path
    path_to_vcl_background_dataset = args.vcl_path
    path_to_intnet_background_dataset = args.intnet_path

    # box flags
    box_flags_map = {
        17 : BoxRenderFlags.LABEL_TOP_AND_BOTTOM_AS_BACKGROUND,    
        21 : BoxRenderFlags.LABEL_DOWN_AS_BACKGROUND,    
        25 : None
    }

    rnd_seed = 1234
    random.seed(rnd_seed)       # this will generate fixed seeds of subcomponents that create the datasets (factory uses random.random() to initialize seeds)
    torch.random.manual_seed(rnd_seed)

    dataset_type = dataset_factory.BoxPoseDatasetType.VERTICAL_2


    # create dataset of 16:9 resolution based on RS2
    if args.train_data_type in ["synthetic","both"]:
        if args.dr is None and args.dz is None and args.dphi is None:
            params = pose_sampler.PoseSamplerParams(
                num_positions = sample_count_per_subdataset,
                rmin = args.rmin,
                rmax = args.rmax,
                zmin = args.zmin,
                zmax = args.zmax,
                look_at_radius = args.lookat,
                up_vector_variance=args.upvecvar
            )
            dsiterators  = dataset_factory.create_rs2_16_9_box_pose_dataset(path_to_corbs_background_dataset,path_to_vcl_background_dataset,
                        path_to_intnet_background_dataset,
                        pose_params = params,
                        box_render_flags=box_flags_map[args.nclasses],                  
                        dataset_type = dataset_type,
                        out_resolution_width = model_params["width"], out_resolution_height = model_params["height"])
        elif args.dr is not None and args.dz is not None and args.dphi is not None:
            params = pose_sampler.PoseSamplerParamsGrid(
                rmin = args.rmin,
                rmax = args.rmax,
                dr = args.dr,
                zmin = args.zmin,
                zmax = args.zmax,
                dz = args.dz,
                look_at_radius = args.lookat,
                up_vector_variance=args.upvecvar,
                dphi = args.dphi
            )
            dsiterators  = dataset_factory.create_rs2_16_9_grid_box_pose_dataset(path_to_corbs_background_dataset,path_to_vcl_background_dataset,
                        path_to_intnet_background_dataset,
                        pose_params = params,
                        box_render_flags=box_flags_map[args.nclasses],                  
                        dataset_type = dataset_type,
                        out_resolution_width = model_params["width"], out_resolution_height = model_params["height"])
        else:
            raise Exception("Not valid")
        
    else:
        dsiterators = list()

    ########

    # real data
    if args.train_data_type in ["real","both"]:
        real_data_params = real_dataloader.DataLoaderParams(\
            root_path=args.real_data_path, device_list=args.device_list,\
            device_repository_path=args.real_data_path, depth_threshold=args.depth_thres, decimation_scale = 4) 
        real_data_iterator = real_dataloader.DataLoad(real_data_params)
        dsiterators.append(real_data_iterator)

    dsiterator = torch.utils.data.ConcatDataset(dsiterators)
    #end of real data
    
    num_workers = args.num_workers
    dataset =  torch.utils.data.DataLoader(dsiterator,\
        batch_size = args.batch_size, shuffle=True,\
        num_workers = num_workers, pin_memory=False)

    ##### TODO VALIDATION DATALOADER HERE #####
    if args.valset_path is not None:
        vdsiterator_params = depthmap_val_dataset.DepthmapDatasetParams(args.valset_path, 0.001, 4)
        vdsiterator = depthmap_val_dataset.DepthmapDataset(vdsiterator_params)

        vdataset = torch.utils.data.DataLoader(vdsiterator,\
            batch_size = args.batch_size, shuffle=True,\
            num_workers = 0, pin_memory=False)

    if args.test_data_path is not None:
        test_iterator_params = depthmap_val_dataset.DepthmapDatasetParams(
            args.test_data_path, 0.001,
            max_len = None,
            number_of_classes = args.nclasses)

        test_iterator = depthmap_val_dataset.DepthmapDataset(test_iterator_params)    
        test_dataset = torch.utils.data.DataLoader(test_iterator,\
            batch_size = args.batch_size, shuffle=False,\
            num_workers = args.num_workers, pin_memory=True)


    ###########################################

    if args.nclasses not in (17,21,25):
        raise Exception("Wrong class number argument ({})".format(args.nclasses))
    else:
        class_w = torch.ones((args.nclasses)).float()


    seg_criterion_1 = nn.NLLLoss2d(weight=class_w, reduction='mean').to(device)
    L2_criterion = nn.MSELoss().to(device)
    L2_norm_criterion = nn.MSELoss().to(device)

    #logging init
    batch_seg_loss = other.AverageMeter()
    batch_heat_loss = other.AverageMeter()
    batch_surface_loss = other.AverageMeter()
    batch_total_loss = other.AverageMeter()
    batch_soft_cor_loss = other.AverageMeter()
    batch_soft_cor_loss_unlabeled = other.AverageMeter()

    frame_index = 0
    for epoch in range(start_epoch, args.epochs):
        
        #init
        seg_loss = 0.0
        heat_loss = 0.0
        surface_loss = 0.0
        total_loss = 0.0
        model.train()

        for batch_id, batch in enumerate(dataset):
            
            if iterations > args.train_duration:
                epoch = args.epochs + 1
                break
            start = time.perf_counter()
            optimizer.zero_grad()
            
            batch_d = batch['depth']

            #forward pass
            if model_name == 'with_normals':
                activs, heat_pred, out, normals = model(batch_d.to(device))
            elif model_name == 'heatmap':
                activs, heat_pred, out = model(batch_d.to(device))
            else:
                activs, out = model(batch_d.to(device))


            real_batch, synth_batch, real_ids, synth_ids = utils.train_utils.split_batch(batch)

            if real_batch:
                out_real = out[real_ids]

            #prepare target
            if synth_batch:
                labels = synth_batch['labels'].float()
                target = labels
                out_synth = out[synth_ids]
            
                #prepare heatmap target
                seg_loss = seg_criterion_1(out_synth, target.squeeze(1).long().to(device))
                


            total_loss = seg_loss
            


            if synth_batch is not None:
                soft_cor_loss = src.supervision.losses.soft_correspondences_loss(
                    torch.exp(out_synth),
                    synth_batch,
                    confidence = 0.0,
                    criterion = L2_criterion,
                    device = device,
                    SVD=args.soft_cor
                )
                if soft_cor_loss is not None:
                    total_loss += args.cor_weight * soft_cor_loss


            #backprop + grad update
            total_loss.backward()
            optimizer.step()


            if synth_batch:
                batch_seg_loss.update(seg_loss.cpu().detach())
            if args.cor_weight != 0.0 and synth_batch is not None and soft_cor_loss is not None:
                batch_soft_cor_loss.update(soft_cor_loss.cpu().detach())
            if model_name == 'heatmap':
                batch_heat_loss.update(heat_loss.cpu().detach())
            if model_name == 'with_normals':
                batch_heat_loss.update(heat_loss.cpu().detach())
                batch_surface_loss.update(surface_loss.cpu().detach())
            batch_total_loss.update(total_loss.cpu().detach())

            if real_batch is not None:
                batch_soft_cor_loss_unlabeled.update(soft_cor_loss_unlabeled.cpu().detach())

            iterations += args.batch_size
            print("Epoch: {}, iteration: {}, learning rate: {}, batch time: {}, Total Loss: {}, Seg Loss: {}, Heat Loss: {}, Surface Loss: {}\n"\
                    .format(epoch, iterations, optimizer.param_groups[0]['lr'], time.perf_counter() - start, batch_total_loss.avg.item(), batch_seg_loss.avg.item(), batch_heat_loss.avg.item(), batch_surface_loss.avg.item()))

            #visualization - Visdom
            if ((iterations) % args.disp_iters) == 0:
                visualizer.append_loss(epoch + 1, iterations, batch_total_loss.avg, "total_loss")
                if real_batch is not None:
                    visualizer.append_loss(epoch + 1, iterations, batch_soft_cor_loss_unlabeled.avg, "soft_cor_loss_unlabeled")
                visualizer.append_loss(epoch + 1, iterations, batch_seg_loss.avg, "segmentation_loss")
                if args.cor_weight != 0.0:
                    visualizer.append_loss(epoch + 1, iterations, batch_soft_cor_loss.avg, "soft_correspondences_loss")
                if model_name == 'heatmap':
                    visualizer.append_loss(epoch + 1, iterations, batch_heat_loss.avg, "heatmap_loss")
                if model_name == 'with_normals':
                    visualizer.append_loss(epoch + 1, iterations, batch_heat_loss.avg, "heatmap_loss")
                    visualizer.append_loss(epoch + 1, iterations, batch_surface_loss.avg, "surface_loss")
            if (iterations % args.visdom_iters) == 0:
                for bidx in range(np.min([batch_d.size()[0], 5])):
                    visualizer.show_seg_map(out[bidx].argmax(0), 'segmentation prediction' + str(bidx))
                    visualizer.show_seg_map(batch_d[bidx], 'input depth'  + str(bidx))
                    if model_name == 'heatmap':
                        visualizer.show_seg_map(heat_gt[bidx], 'heatmap gt' + str(bidx))
                        visualizer.show_seg_map(heat_pred[bidx], 'heatmap prediction' + str(bidx))
                    if model_name == 'with_normals':
                        visualizer.show_seg_map(heat_pred[bidx], 'heatmap prediction' + str(bidx))
                        visualizer.show_seg_map(heat_gt[bidx], 'heatmap gt' + str(bidx))
                        visualizer.show_normals(normals[bidx], 'normals prediction' + str(bidx))
                        visualizer.show_normals(normals_target[bidx].float(), 'normals gt' + str(bidx))

        #validation
        if args.valset_path is not None:
            with torch.no_grad():
                model.eval()
                confidence_threshold = args.confidence_threshold
                frame_index = 0
                total_iou = 0
                for vbatch_id, vbatch in enumerate(vdataset):
                    #resize input
                    vbatch_d = vbatch['depth']
                    
                    #prepare target
                    vlabels = vbatch['labels'].float()

                    activs, out = model(vbatch_d.to(device))

                    batch_size = vbatch_d.shape[0]

                    confidence_t, labels_pred_t = out.max(dim = 1, keepdim = True)       
                    confidence_t = torch.exp(confidence_t)                          # convert log probability to probability
                    labels_pred_t [confidence_t < confidence_threshold] = 0  # uncertain classs

                    frame_index += batch_size
                    sample_iou, mask = metrics.jaccard(labels_pred_t.cpu().float(), vlabels, args.nclasses)
                    total_iou = torch.sum(sample_iou*mask.float(), dim = -1) / mask.sum(dim = -1).float()

                print("Epoch: {}, Average IoU: {}\n"\
                    .format(epoch, total_iou.mean()))

                visualizer.append_loss(epoch + 1, iterations, total_iou.mean(), "average IoU", mode='val')

        if args.test_data_path is not None:
            total_iou = 0
            frame_index = 0
            bar = tqdm(total = test_iterator.__len__())
            with torch.no_grad():
                model.eval()
                confidence_threshold = args.confidence_threshold
                for batch_id, batch in enumerate(test_dataset):
                    vbatch_d = batch['depth']
                    
                    #prepare target
                    vlabels = batch['labels'].float()

                    activs, out = model(vbatch_d.to(device))

                    batch_size = vbatch_d.shape[0]

                    confidence_t, labels_pred_t = out[index].max(dim = 1, keepdim = True)      
                    confidence_t = torch.exp(confidence_t)                          # convert log probability to probability
                    labels_pred_t [confidence_t < confidence_threshold] = 0  # uncertain classs
                    
                    frame_index += batch_size
                    sample_iou, mask = metrics.jaccard(labels_pred_t.cpu().float(), vlabels, args.nclasses)
                    total_iou = torch.sum(sample_iou*mask.float(), dim = -1) / mask.sum(dim = -1).float()

                print("Epoch: {}, Test Average IoU: {}\n"\
                    .format(epoch, total_iou.mean()))

                visualizer.append_loss(epoch + 1, iterations, total_iou.mean(), "average test IoU", mode='val')


        #save model params
        src.utils.save_checkpoint({
            'nclasses'  : args.nclasses,
            'ndf' : args.ndf,
            'epoch': epoch,            
            'iterations': iterations,
            'batch_size': args.batch_size,
            'model_name': model_name,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),            
        }, epoch, name = args.name)

        torch.cuda.empty_cache()
        if epoch > args.epochs:
            break