import os, glob, shutil
import urllib.request as r
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch.utils.data import random_split
from torchvision.transforms import functional as F, transforms as T

def download_dataset(path_to_download, dataset_name = "vehicle"): 
    
    assert dataset_name == "vehicle", f"Iltimos vehicle nomi bilan kiriting!"
    if dataset_name == "vehicle": url = "kaggle datasets download -d killa92/vehicle-detection-dataset"
    
    # Check if is already exist 
    if os.path.isfile(f"{path_to_download}/{dataset_name}.csv") or os.path.isdir(f"{path_to_download}/{dataset_name}"): 
        print(f"Dataset allaqachon yuklab olingan. {path_to_download}/{dataset_name} papkasini ni tekshiring.\n"); 

    # If data doesn't exist in particular folder
    else: 
        ds_name = url.split("/")[-1] 
        # Download the dataset
        print(f"{ds_name} yuklanmoqda...")
        os.system(f"{url} -p {path_to_download}")
        shutil.unpack_archive(f"{path_to_download}/{ds_name}.zip", extract_dir=f"{path_to_download}/{dataset_name}")
        os.remove(f"{path_to_download}/{ds_name}.zip")
        print(f"Tanlangan dataset {path_to_download}/{dataset_name} papkasiga yuklab olindi!")
    
    return f"{path_to_download}/{dataset_name}"

#make a Custom dataset class 
class VehicleDataset(Dataset):  
    def __init__(self, root, n_classes, transformations = None, ):
        super().__init__()
        
        self.transformations = transformations
        #self.img_paths = glob.glob(f"{root}/images/*") 
        #rasm nomi va label bir xil nomda bolgani sababli .txt -> .png ga otkazib rasmni oqib olishimiz mumkun. label pathni ozi yetadi
        self.labels = sorted(glob.glob(f"{root}/labels/*")) 
        self.img_width, self.img_height = 720, 480 #orginal img size
        self.num_classes = n_classes #fast rcnn da doim +1. sababi bittasi background uchun boladi. bizda faqat car class +1 thats why 2 boldi
        self.data = {} #key:img_path, value:bboxes 

        for i, label_path in enumerate(self.labels): 
            img_path = label_path.replace("txt", "png").replace("labels","images")
            bbox_coordinates = []
            bboxes = open(label_path, mode='r').read().split("\n")[:-1] #rasmdagi har bir obyektni bboxini nextline boyicha olish
            if len(bboxes) < 1: continue #rasmda object umuman yoq holatda 
            
            #har bir boxga kiradi va yolo2cv2 funksiyasini chaqirib -> u esa har bir box x,y,w,h cordinatani olib
            #opencv formatga formula orqali o'tkazadi otkazadi va qaytaradi
            for bbox in bboxes: bbox_coordinates.append(self.yolo2cv(bbox))

            self.data[img_path] = bbox_coordinates
    
    def yolo2cv(self, bbox): 
        # In function calls, the asterisk (*) can be used to unpack a list or tuple into individual arguments
        #Bu yerda list comprehensiondan chiqqan 4 ta qiymatni x, y, w, h argument sifatida jonatyapti.
        return self.get_coordinates(*[float(bb) for bb in bbox.split(" ")[1:]])

    def get_coordinates(self, x, y, w, h):
        """ This function formula converts [0, 1] range YOLO format to OpenCV format"""
        return [int((x - (w / 2)) * self.img_width), 
                int((y - (h / 2)) * self.img_height), 
                int((x + (w / 2)) * self.img_width), 
                int((y + (h / 2)) * self.img_height)]
    
        #l - left, r -right, t - top, b - bottom 
        #  l = int((x - (w / 2)) * self.img_width)
        #  r = int((x + (w / 2)) * self.img_width)
        #  t = int((y - (h / 2)) * self.img_height)
        #  b = int((y + (h / 2)) * self.img_height)

        #  return [l, t, r, b]

    
    def get_area(self, bboxes): 
        """bu funksiya faster rcnn ni areani hisoblash formulasi"""
        #bboxes list ichida yana [],[] kichik obyekt joylashgan coor listlar mavjud 
        #[:,3] - kotta listdagi barchasi va har bir kichik listdagi 3-indexdagi element digani
        return (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
    

    def create_target(self, bboxes, labels, iscrowd, image_id, area): 
        """Bu funksiya FasterRCNN uchun maxsus data structurasi hisoblanadi,
        target dictionary ochganda ichidagi keys lar bboxes, labels, iscrowd, image_id, area bolishi kerak
        aks holda typoo da xato bo'lsa training paytida ishlamaydi"""
        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels
        target['iscrowd'] = iscrowd
        target['image_id'] = image_id
        target['area'] = area
        return target
        
    def __len__(self):  return len(self.data)
    
    def __getitem__(self, index):
        img_path = list(self.data.keys())[index]
        image = Image.open(img_path).convert('RGB')
        bboxes = torch.as_tensor(self.data[img_path], dtype=torch.float32) 
        labels = torch.ones((len(bboxes),), dtype = torch.int64) #obyekt sonini aniqlash uchun 3 ta bolsa 1 lar 3 ta obyekt
        iscrowd = torch.zeros((len(bboxes),), dtype = torch.int64)
        image_id =  torch.tensor([index])
        area = self.get_area(bboxes)
        
        target = self.create_target(bboxes, labels, iscrowd, image_id, area)

        if self.transformations: 
            image, target = self.transformations(image, target)

        return image, target
    

def custom_collate_fn(batch): 
    """Bu funksiya faster rcnn uchun. Yani har bir rasmga mutanosib targetni chiqarib berish
    Natija: rasmlar, bboxes, labels, iscrowd, image_id, are = batch[0]""" 
    return tuple(zip(*batch))

def get_dataloaders(root, n_classes, tfs, bs,  split = [0.8, 0.1, 0.1], ns = 0):
    
    ds = VehicleDataset(root = root, n_classes = n_classes, transformations = tfs)
    
    train_data, valid_data, test_data  = random_split(dataset = ds, lengths = split, 
                                               generator = torch.Generator().manual_seed(13)) #80% for training....

    print(f"Train data points:{len(train_data)}   |  Valid data points:{len(valid_data)}    |    Test data points: {len(test_data)}")
   
    tr_dl = DataLoader(dataset = train_data, batch_size = bs, collate_fn = custom_collate_fn, shuffle = True, num_workers = ns)
    val_dl = DataLoader(dataset = valid_data, batch_size = bs, collate_fn = custom_collate_fn, shuffle = False, num_workers = ns)
    test_dl = DataLoader(dataset = test_data, batch_size = 1, collate_fn = custom_collate_fn, shuffle = False, num_workers = ns)

    print("Batch size:", bs)
    print(f"Train size of batches:{len(tr_dl)}   |  Valid size of batches:{len(val_dl)}    |    Test size of batches: {len(test_dl)}\n")
    
    return tr_dl, val_dl, test_dl, ds.num_classes



class Compose: 

    def __init__(self, transforms): 
        self.transforms = transforms

    def __call__(self, image, target): 

        for tr in self.transforms: image, target = tr(image, target)
        
        return image, target
    
class PIL2Tensor(torch.nn.Module): 
    
    def forward(self, image, target): 
        image = F.pil_to_tensor(image)
        return image, target

class Normalize(torch.nn.Module): 
    
    def forward(self, image, target): 
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] #ImagaNet values
        image = F.normalize(tensor=image, mean=mean, std=std)
        return image, target
    
class ConvertImageDtype(torch.nn.Module): 

    def __init__(self,dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, image, target): 
        image = F.convert_image_dtype(image=image, dtype=self.dtype)
        return image, target
    

def get_transformations(): 
    transformations = []
    transformations.append(PIL2Tensor())
    transformations.append(ConvertImageDtype(dtype = torch.float))
    transformations.append(Normalize())
    
    return Compose(transforms = transformations )


if __name__ == "__main__":
    tfs = get_transformations()

    dataset_path = download_dataset("data")
    tr_dl, val_dl, test_dl, num_classes = get_dataloaders(root=dataset_path, tfs=tfs, bs=10)
