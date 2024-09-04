import argparse
import os
import random
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from dataset.medical_few import MedDataset
from VLM.clip import create_model
from VLM.tokenizer import tokenize
from VLM.adapter import CLIP_Inplanted
from PIL import Image
from sklearn.metrics import roc_auc_score, precision_recall_curve, pairwise
from loss import FocalLoss, BinaryDiceLoss
from utils import augment, cos_sim, encode_text_with_prompt_ensemble
from prompt import REAL_NAME
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

from labeling_model import predict_label
from report_llm import generate_report

import warnings
warnings.filterwarnings("ignore")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "mps")

CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Histopathology':-3}
CLASS_INDEX_INV = {3:'Brain', 2:'Liver', 1:'Retina_RESC', -1:'Retina_OCT2017', -3:'Histopathology'}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(path):
    # Default values
    model_name = 'ViT-L-14-336'
    pretrain = 'openai'
    obj = 'Brain'
    data_path = './data/'
    batch_size = 1
    img_size = 240
    save_path = './checkpoint/few-shot/'
    epoch = 50
    learning_rate = 0.0001
    features_list = [6, 12, 18, 24]
    seed = 111
    shot = 4
    iterate = 0

    """parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--path', type=str, default='')
    args = parser.parse_args()
    test_img = args.path"""
    test_img = path

    setup_seed(seed)
    print("Starting Inference")
    # Fixed feature extractor
    clip_model = create_model(model_name=model_name, img_size=img_size, device=device, pretrained=pretrain, require_pretrained=True)
    clip_model.eval()
    print("Model loaded succesfully...Loading adapter layers")
    model = CLIP_Inplanted(clip_model=clip_model, features=features_list).to(device)
    model.eval()

    checkpoint = torch.load(os.path.join(f'{save_path}', f'{obj}.pth'), map_location=torch.device('mps'))
    model.seg_adapters.load_state_dict(checkpoint["seg_adapters"])
    model.det_adapters.load_state_dict(checkpoint["det_adapters"])
    print("Loaded Adapters!\nSetting up for few shot....")
    for name, param in model.named_parameters():
        param.requires_grad = True

    # Optimizer for only adapters
    seg_optimizer = torch.optim.Adam(list(model.seg_adapters.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    det_optimizer = torch.optim.Adam(list(model.det_adapters.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    # Load dataset and loader
    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    test_dataset = MedDataset(data_path, obj, img_size, shot, iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    # few-shot image augmentation
    augment_abnorm_img, augment_abnorm_mask = augment(test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask)
    augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

    augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
    augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)

    augment_fewshot_label = torch.cat([torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0)

    train_dataset = torch.utils.data.TensorDataset(augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)


    # memory bank construction
    support_dataset = torch.utils.data.TensorDataset(augment_normal_img)
    support_loader = torch.utils.data.DataLoader(support_dataset, batch_size=1, shuffle=True, **kwargs)


    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()
    loss_bce = torch.nn.BCEWithLogitsLoss()


    # text prompt
    with torch.cuda.amp.autocast(), torch.no_grad():
        text_features = encode_text_with_prompt_ensemble(clip_model, REAL_NAME[obj], device)

    best_result = 0

    seg_features = []
    det_features = []
    for image in support_loader:
        image = image[0].to(device)
        with torch.no_grad():
            _, seg_patch_tokens, det_patch_tokens = model(image)
            seg_patch_tokens = [p[0].contiguous() for p in seg_patch_tokens]
            det_patch_tokens = [p[0].contiguous() for p in det_patch_tokens]
            seg_features.append(seg_patch_tokens)
            det_features.append(det_patch_tokens)
    seg_mem_features = [torch.cat([seg_features[j][i] for j in range(len(seg_features))], dim=0) for i in range(len(seg_features[0]))]
    det_mem_features = [torch.cat([det_features[j][i] for j in range(len(det_features))], dim=0) for i in range(len(det_features[0]))]

    print("Model ready for inference!\nPassing the image..")
    result = test(model, test_img, text_features, seg_mem_features, det_mem_features)
    print("\n\nFINAL", result)

    #labels = predict_label()
    #report = generate_report(labels)
    #print(report)
    return "Under construction"



def test(model, test_img, text_features, seg_mem_features, det_mem_features):
    # Default values
    model_name = 'ViT-L-14-336'
    pretrain = 'openai'
    obj = 'Brain'
    data_path = './data/'
    batch_size = 1
    img_size = 240
    save_path = './ckpt/few-shot/'
    epoch = 50
    learning_rate = 0.0001
    features_list = [6, 12, 18, 24]
    seed = 111
    shot = 4
    iterate = 0
    gt_list = []
    gt_mask_list = []

    # Load and preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    image = Image.open(test_img).convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)

    det_image_scores_zero = []
    det_image_scores_few = []

    seg_score_map_zero = []
    seg_score_map_few = []
    
    print("Done Precessing of the image!")
    with torch.no_grad(), torch.cuda.amp.autocast():
        _, seg_patch_tokens, det_patch_tokens = model(image)
        seg_patch_tokens = [p[0, 1:, :] for p in seg_patch_tokens]
        det_patch_tokens = [p[0, 1:, :] for p in det_patch_tokens]

        if CLASS_INDEX[obj] > 0:
            # few-shot, seg head
            anomaly_maps_few_shot = []
            for idx, p in enumerate(seg_patch_tokens):
                cos = cos_sim(seg_mem_features[idx], p)
                height = int(np.sqrt(cos.shape[1]))
                anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                        size=img_size, mode='bilinear', align_corners=True)
                anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
            score_map_few = np.sum(anomaly_maps_few_shot, axis=0)
            seg_score_map_few.append(score_map_few)

            # zero-shot, seg head
            anomaly_maps = []
            for layer in range(len(seg_patch_tokens)):
                seg_patch_tokens[layer] /= seg_patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * seg_patch_tokens[layer] @ text_features).unsqueeze(0)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=img_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :]
                anomaly_maps.append(anomaly_map.cpu().numpy())
            score_map_zero = np.sum(anomaly_maps, axis=0)
            seg_score_map_zero.append(score_map_zero)

            # Visualize the original image and heatmaps
            print("Anomaly map generated!")
            visualize_anomaly(image, score_map_few, "few-shot")
            visualize_anomaly(image, score_map_zero, "zero-shot")

        else:
            # few-shot, det head
            anomaly_maps_few_shot = []
            for idx, p in enumerate(det_patch_tokens):
                cos = cos_sim(det_mem_features[idx], p)
                height = int(np.sqrt(cos.shape[1]))
                anomaly_map_few_shot = torch.min((1 - cos), 0)[0].reshape(1, 1, height, height)
                anomaly_map_few_shot = F.interpolate(torch.tensor(anomaly_map_few_shot),
                                                        size=img_size, mode='bilinear', align_corners=True)
                anomaly_maps_few_shot.append(anomaly_map_few_shot[0].cpu().numpy())
            anomaly_map_few_shot = np.sum(anomaly_maps_few_shot, axis=0)
            score_few_det = anomaly_map_few_shot.mean()
            det_image_scores_few.append(score_few_det)

            # zero-shot, det head
            anomaly_score = 0
            for layer in range(len(det_patch_tokens)):
                det_patch_tokens[layer] /= det_patch_tokens[layer].norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * det_patch_tokens[layer] @ text_features).unsqueeze(0)
                anomaly_map = torch.softmax(anomaly_map, dim=-1)[:, :, 1]
                anomaly_score += anomaly_map.mean()
            det_image_scores_zero.append(anomaly_score.cpu().numpy())

            # Visualize the original image and heatmaps
            visualize_anomaly(image, anomaly_map_few_shot, "few-shot")
            visualize_anomaly(image, anomaly_map, "Zero-shot")
    
def visualize_anomaly(image_tensor, anomaly_map, shot):
    # Convert tensor to numpy array
    image_np = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Normalize the anomaly map for visualization
    anomaly_map = anomaly_map.squeeze()
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    # Apply a heatmap to the anomaly map
    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlayed_image = cv2.addWeighted(image_np, 0.5, heatmap, 0.5, 0)

    # Save the result using OpenCV
    cv2.imwrite('anomaly_heatmap_overlay.png', cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR))
    print("Anomaly image saved locally..")
    # Display the original image and the heatmap
    """plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(heatmap, alpha=0.5)
    plt.title('Anomaly Heatmap '+shot)

    plt.show()"""
