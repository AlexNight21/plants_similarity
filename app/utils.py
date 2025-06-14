from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import torch
from torch.nn.functional import cosine_similarity
import pickle
import numpy as np
import json


def get_embeddings_from_img(image_path, image_transforms, device, embedd_model):

    # get image features
    try:
        image = Image.open(image_path).convert("RGB")
        image = image_transforms(image).to(device)

        with torch.inference_mode():
            image_feat = embedd_model(image.unsqueeze(0)).squeeze(0).view(-1)
            image_feat = image_feat.cpu().numpy()
        
        return image_feat
    
    except Exception as e:
        print(f"[ERROR] Error processing image {image_path}: {e}")


def compute_images_similarity(img_feature, imgs_features_lst):
        
    features_array = torch.tensor(np.vstack(imgs_features_lst))
    img_feature = torch.tensor(img_feature).unsqueeze(0)
    similarity = cosine_similarity(img_feature, features_array, dim=1)
    
    cos_distance = 1 - similarity
    
    return cos_distance


def get_sorted_images(imgs_paths_lst, cos_distance, num_showed_imgs):
    
    idxs = cos_distance.argsort()[:num_showed_imgs]
    
    imgs_lst = []
    
    for idx in idxs:
        data_dict = {
            "image_path": imgs_paths_lst[idx],
            "similarity_score": round(cos_distance[idx].item(), 4),
        }

        imgs_lst.append(data_dict)
        
    return imgs_lst
        

def save_json_data(dict_data, report_path):
    with open(report_path, 'w') as f:
        json.dump(dict_data, f)


def get_similar_images(
    image_path,
    features_data_path,
    sim_imgs_num,
    report_path,
):
    
    try:
    
        # set model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
        embedd_model = torch.nn.Sequential(*list(model.children())[:-1])
        
        # set imagenet transforms
        image_transforms = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        
        # read imgs features data
        with open(features_data_path, "rb") as f:
            imgs_features = pickle.load(f)
        
        imgs_paths_lst = []
        imgs_features_lst = []
        
        for feat in imgs_features:
            imgs_paths_lst.append(feat["path"])
            imgs_features_lst.append(feat["features"])
        
        random_img_feature = get_embeddings_from_img(
            image_path=image_path,
            image_transforms=image_transforms,
            device=device,
            embedd_model=embedd_model,
        )
        
        cos_distance = compute_images_similarity(
            img_feature=random_img_feature,
            imgs_features_lst=imgs_features_lst,
        )
        
        imgs_lst = get_sorted_images(
            imgs_paths_lst=imgs_paths_lst,
            cos_distance=cos_distance,
            num_showed_imgs=sim_imgs_num,
        )

        save_json_data(dict_data=imgs_lst, report_path=report_path)
        
        print("[INFO] Success!")
        
        return True
    
    except Exception as e:
        print(f"[ERROR] An error occurred while data processing!\n{e}")
