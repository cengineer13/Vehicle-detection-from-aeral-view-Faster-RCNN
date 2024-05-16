import torch, argparse, random
import cv2, numpy as np
from PIL import Image
from torchvision import transforms as T
from dataset import get_transformations
import streamlit as st
from torchvision.models.detection import fasterrcnn_resnet50_fpn

def run(args): 

    #load our best model
    model = load_model(model_path=f"{args.model_files}/{args.dataset_name}_best_model.pt")
    print(f"Train qilingan model {args.model_name} muvaffaqiyatli yuklab olindi.!")

    #img for prediction
    st.title("Vehicle object detection from aerial view by Faster RCNN algorithm")
    img_path = st.file_uploader("Rasmni yuklang...")

    tfs = get_transformations()
    org_img, pred_img, n_obj = predict(model, img_path, tfs, args.device, threshold=args.threshold) if img_path else predict(model, args.test_img, tfs, args.device, threshold=args.threshold)
    
    def center_text(text: str):
            st.markdown(f"<h3 style='text-align: center; color: black;'>{text}</h3>", unsafe_allow_html=True)
    
     # Create two columns
    col1, col2 = st.columns(2)
    with col1:
        # Use the function to center your text
        center_text("Before prediction")
        st.image(org_img, caption="Orginal image")


    with col2:
        center_text("After prediction")
        st.image(pred_img, caption=f"{n_obj} {args.dataset_name}s are detected!")
        


def load_model(model_path):
    m = torch.load(model_path)
    return m.eval()

def predict(model, img_path, tfs, device, threshold):  
    org_img = Image.open(img_path).convert('RGB').resize((720,480))
    cv_img = np.array(org_img)
    target = {} #just for transformations
    tensor_img, _ = tfs(org_img, target)
    with torch.no_grad(): 
        predictions = model(tensor_img.unsqueeze(0).to(device))
    obj_count = 0
    for i, (boxes, scores, labels) in enumerate(zip(predictions[0]["boxes"], predictions[0]["scores"], predictions[0]["labels"])):
        
        if scores > threshold:
            obj_count += 1
            r, g, b = [random.randint(0, 255) for _ in range(3)]
            x, y, w, h = [round(b.item()) for b in boxes]
            cv2.rectangle(img = cv_img, pt1 = (x, y), pt2 = (w, h), color = (r, g, b), thickness = 3)
    
    return org_img, cv_img, obj_count 
    
if __name__ == "__main__":
   
    parser = argparse.ArgumentParser(description="Vehicle object detection Faster RCNN DEMO")
    #Add arguments (- va -- option value oladigon, type - qaysi turni olish )
    parser.add_argument("-mf", "--model_files", type=str, default="data/model_files", 
                        help="Train qilingan model va boshqa parametr fayllar uchun yo'lak")
    parser.add_argument("-dn", "--dataset_name", type=str, default = "vehicle", help="Dataset nomi")
    parser.add_argument("-mn", "--model_name", type=fasterrcnn_resnet50_fpn, default=fasterrcnn_resnet50_fpn, help="Trained bo'lgan object detection algorithm")
    parser.add_argument("-dv", "--device", type=str, default="cuda:0", help="Train qilish qurilmasi GPU yoki CPU")
    parser.add_argument("-th", "--threshold", type=float, default=0.7, help="Threshold for inference")
    parser.add_argument("-ti", "--test_img", default="data/test_images/many2.jpg", help="Path for image to predict unseen image")

    args = parser.parse_args()
    run(args=args)
    