import random, numpy as np, cv2
import os
from matplotlib import pyplot as plt
from torchvision.transforms import transforms as T

def tensor_2_im(t, t_type = "rgb", inv_trans = False):
    
    assert t_type in ["rgb", "gray"], "Rasm RGB yoki grayscale ekanligini aniqlashtirib bering."
    gray_tfs = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb_tfs  = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    invTrans = gray_tfs if t_type == "gray" else rgb_tfs 
    
    return (invTrans(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8) if inv_trans else (t * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def visualize(data, n_ims, rows, cmap = None, data_type = None, save_folder=None):
    
    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(data) - 1) for _ in range(n_ims)]
    for idx, indeks in enumerate(indekslar):
        
        rasm, target = data[indeks]
        org_rasm = tensor_2_im(rasm, inv_trans = True)
        obj_hisob = 0
        for i, bbox in enumerate(target["boxes"]):
            x, y, w, h = [int(t.item()) for t in bbox]
            obj_hisob += 1
            cv2.rectangle(img = org_rasm, pt1 = (x, y), pt2 = (w, h), color = (200, 0, 255), thickness = 3)
        plt.subplot(rows, n_ims // rows, idx + 1)
        plt.title(f"Rasmda {obj_hisob}ta object bor.");
        plt.imshow(org_rasm)
        plt.axis("off")  

    #if saving folder is not available
    os.makedirs(f"{save_folder}", exist_ok=True)   
    plt.savefig(f"{save_folder}/{data_type}-object detection examples.png")
    print(f"{data_type} datasetdan namunalar {save_folder} papkasiga yuklandi...")
    #plt.show()
    print("---------------------------------------------------------------------")
    
      
        
            

            