import argparse, os
import torch
import torch.nn as nn
from dataset import download_dataset, get_transformations, get_dataloaders
from vis_utils import visualize
from train import train_one_epoch, evaluate
from inference import inference
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def run(args): 
    
    assert args.down_path == "data", "Iltimos data papkasini kiriting"
    # 1 - download dataset
    root_ds = download_dataset(path_to_download = args.down_path, dataset_name = args.dataset_name)
    
    # 2 - Get faster rcnn data specialized manual coded transformation 
    tfs = get_transformations()
    
    # 3 - get dataloaders 
    tr_dl, val_dl, test_dl, num_classes = get_dataloaders(root=root_ds, n_classes = args.n_classes,
                                                          tfs=tfs, bs=args.batch_size, split=args.split_ratio, 
                                                          ns=args.num_workers)

    # 4 - save visualized examples from dataset (not dataloaders)
    for dl, data_type in zip([tr_dl, val_dl, test_dl], ['train', 'val', 'test']):
        visualize(data=dl.dataset, n_ims=args.n_imgs, rows = args.rows, cmap='rgb', 
                     data_type=data_type, save_folder=args.vis_path)

    # 5 - train and validation process
    model = fasterrcnn_resnet50_fpn(weights = "DEFAULT")
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor.cls_score = torch.nn.Linear(in_features, num_classes)
    model.roi_heads.box_predictor.bbox_pred = torch.nn.Linear(in_features, num_classes * 4)
    model.to(args.device)
    optimizer = args.optimizer(model.parameters(), lr = args.learning_rate, momentum = 0.9, weight_decay = 0.0005)
    # a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 3, gamma = 0.1)
    
    print('...................TRAIN JARAYONI BOSHLANDI!.........................')
    for epoch in range(args.epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, tr_dl, args.device, epoch, print_freq = 10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, val_dl, device = args.device)

    os.makedirs(f"{args.model_files}", exist_ok = True)
    torch.save(model, f"{args.model_files}/{args.dataset_name}_best_model.pt")
    print('...................TRAIN JARAYONI YAKUNLANDI!.........................\n')
    
    #6 - Inference part 
    print('...................INFERENCE JARAYONI BOSHLANDI!......................')
    m = torch.load(f"{args.model_files}/{args.dataset_name}_best_model.pt")
    inference(model = m, ts_dl = test_dl, num_ims = args.n_imgs, rows = args.rows, 
              device=args.device, save_folder=args.vis_path, threshold = args.threshold, cmap = "rgb")
    print('...................INFERENCE JARAYONI YAKUNLANDI!......................\n')
    

if __name__ == "__main__": 
  
    parser = argparse.ArgumentParser(description="Vehicle objcet detection Faster RCNN")
    parser.add_argument("-dp", "--down_path", type=str, default="data", help="Datasetni yuklash uchun path")
    parser.add_argument("-dn", "--dataset_name", type=str, default = "vehicle", help="Dataset nomi")
    parser.add_argument("-bs", "--batch_size", type=int, default=10, help="Batch size")
    parser.add_argument("-sp", "--split_ratio", type=list, default=[0.8,0.1,0.1], help="Dataset train val test split ration. ration must be equal to 1")
    parser.add_argument("-nw", "--num_workers", type=int, default= 4, help="Number of workers loading data")
    parser.add_argument("-n_im", "--n_imgs", type=int, default = 20, help="Number of images for plotting")
    parser.add_argument("-r", "--rows", type=int, default = 5, help="Number of rows for plotting as subplots")
    parser.add_argument("-vs", "--vis_path", type=str, default="data/plots", help="Vizualizations graph, plotlarni saqlash uchun yo'lak")
    parser.add_argument("-nc", "--n_classes", type=int, default=2, help="Number of Object detection classes default 2 with background")  
    #parser.add_argument("-mn", "--model_name", type=fasterrcnn_resnet50_fpn, default = fasterrcnn_resnet50_fpn, help="Pretrained bo'lgan pytorch segmentation model nomi")  
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-op", "--optimizer", type=torch.optim.SGD, default=torch.optim.SGD, help="Optimizer function")
    parser.add_argument("-ep", "--epochs", type=int, default=15, help="Epochlar soni")
    parser.add_argument("-dv", "--device", type=str, default="cuda:0", help="Train qilish qurilmasi GPU yoki CPU")
    parser.add_argument("-mf", "--model_files", type=str, default="data/model_files", 
                        help="Train qilingan model va boshqa parametr fayllar uchun yo'lak")
    parser.add_argument("-th", "--threshold", type=float, default=0.5, help="Threshold for inference")

    args = parser.parse_args()
    run(args)