import random, os
import torch
import cv2
import matplotlib.pyplot as plt 
from vis_utils import tensor_2_im

def inference(model, ts_dl, num_ims, rows, device, save_folder, threshold = 0.5, cmap = None):
    
    assert cmap in ["rgb", "gray"], "Rasmni oq-qora yoki rangli ekanini aniqlashtirib bering!"
    if cmap == "rgb": cmap = "viridis"
    
    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(ts_dl) - 1) for _ in range(num_ims)]
    
    for idx, indeks in enumerate(indekslar):
        im, _ = ts_dl.dataset[indeks]
        with torch.no_grad(): predictions = model(im.unsqueeze(0).to(device))
        img = tensor_2_im(im, inv_trans = True)
        obj_count = 0
        for i, (boxes, scores, labels) in enumerate(zip(predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"])):
            if scores > threshold:
                obj_count += 1
                r, g, b = [random.randint(0, 255) for _ in range(3)]
                x, y, w, h = [round(b.item()) for b in boxes]
                cv2.rectangle(img = img, pt1 = (x, y), pt2 = (w, h), color = (r, g, b), thickness = 2)
        plt.subplot(rows, num_ims // rows, idx + 1)
        plt.title(f"{obj_count} vehicles are detected!")
        plt.axis("off")
        plt.imshow(img); 
    #if saving folder is not available
    os.makedirs(f"{save_folder}", exist_ok=True)   
    plt.savefig(f"{save_folder}/inference_results.png")
    print(f"Inference natija {save_folder} papkasiga tushdi..")
    plt.clf(); 
    #plt.show()
            