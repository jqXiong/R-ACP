# /R-ACP/Camera_calibration/main-multi-gpu.py
import cv2
import torch
import torch.nn as nn
import torchreid
from torchvision import transforms
import numpy as np
import os
import xml.etree.ElementTree as ET
from scipy.stats import entropy

# 设置路径参数
base_path = '/home/agou/Desktop/R-ACP/Data/Wildtrack/Image_subsets'
output_image_base_dir = "./24-JSAC/Re_ID_Test/match_image"
output_xml_dir = "./24-JSAC/Re_ID_Test/match_log"
frame_start = 0
frame_end = 2000
frame_step = 5

class PersonDetector:
    def __init__(self, device):
        self.device = device
        with torch.cuda.device(self.device):
            # 添加 skip_validation=True 防止 YOLOv5 烦人的自动更新报错
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, skip_validation=True)
            self.model.to(self.device)
            self.model.eval()

    def detect(self, image):
        with torch.cuda.device(self.device):
            results = self.model(image)
        return results.xyxy[0].cpu().numpy()

class FeatureExtractor:
    def __init__(self, device):
        self.device = device
        with torch.cuda.device(self.device):
            self.extractor = torchreid.utils.FeatureExtractor(
                model_name='osnet_x1_0',
                model_path='osnet_x1_0_imagenet.pth', # 确保这个权重文件在你当前运行目录下
                device=f'cuda:{self.device}' if torch.cuda.is_available() else 'cpu'
            )

    def extract(self, image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0).to(self.device)
        with torch.cuda.device(self.device):
            features = self.extractor(image)
        return features[0].cpu().detach().numpy()

    def compute_communication_cost(self, features_list):
        all_features = np.hstack(features_list)
        hist, _ = np.histogram(all_features, bins=1024, range=(-10, 10), density=True)
        comm_cost = entropy(hist, base=2)
        return comm_cost

def get_image_path(camera_id, frame_id):
    frame_str = f'{frame_id:08d}.png'
    return os.path.join(base_path, f'C{camera_id}', frame_str)

def get_foot_center(box):
    x_center = (box[0] + box[2]) / 2
    y_bottom = box[3]
    return (int(x_center), int(y_bottom))

def create_xml_log(camera_id, frame_id, foot_centers, output_dir, match_camera_id):
    folder_path = os.path.join(output_dir, f'camera{camera_id}_match_camera{match_camera_id}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    root = ET.Element("MatchLog")
    root.set("CameraID", str(camera_id))
    root.set("FrameID", str(frame_id))

    for person_id, (x, y) in foot_centers:
        person_elem = ET.SubElement(root, "Person")
        person_elem.set("ID", str(person_id))
        point_elem = ET.SubElement(person_elem, "FootCenter")
        point_elem.set("x", str(x))
        point_elem.set("y", str(y))

    tree = ET.ElementTree(root)
    output_path = os.path.join(folder_path, f"match_log_frame{frame_id}.xml")
    tree.write(output_path)

# 注意：修改了 main 函数的参数，接收分配给该 GPU 的帧列表 (frames_to_process)
def main_worker(reference_camera_id, unknown_camera_id, frames_to_process, N_threshold, gpu_id):
    torch.cuda.set_device(gpu_id)
    print(f"[GPU {gpu_id}] 开始处理 {len(frames_to_process)} 帧...")
    detector = PersonDetector(gpu_id)
    extractor = FeatureExtractor(gpu_id)

    total_communication_cost = 0

    for frame_id in frames_to_process:
        image1_path = get_image_path(reference_camera_id, frame_id)
        image2_path = get_image_path(unknown_camera_id, frame_id)

        if not os.path.exists(image1_path) or not os.path.exists(image2_path):
            continue

        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        boxes1 = detector.detect(image1)
        boxes2 = detector.detect(image2)

        features1 = [extractor.extract(image1[int(box[1]):int(box[3]), int(box[0]):int(box[2])]) for box in boxes1 if int(box[3]) > int(box[1]) and int(box[2]) > int(box[0])]

        if features1:
            comm_cost = extractor.compute_communication_cost(features1)
            total_communication_cost += comm_cost

            features2 = [extractor.extract(image2[int(box[1]):int(box[3]), int(box[0]):int(box[2])]) for box in boxes2 if int(box[3]) > int(box[1]) and int(box[2]) > int(box[0])]

            def compute_similarity(feature1, feature2):
                return np.linalg.norm(feature1 - feature2)

            matches = []
            for i, feat1 in enumerate(features1):
                for j, feat2 in enumerate(features2):
                    dist = compute_similarity(feat1, feat2)
                    matches.append((i, j, dist))

            matches = sorted(matches, key=lambda x: x[2])
            top_matches = matches[:N_threshold]

            print(f'[GPU {gpu_id}] Frame {frame_id}: Top {N_threshold} matches = {top_matches}')
            print(f'[GPU {gpu_id}] Frame {frame_id}: Comm cost = {comm_cost:.4f} bits')

    if len(frames_to_process) > 0:
        avg_cost = total_communication_cost / len(frames_to_process)
        print(f'[GPU {gpu_id}] 处理完毕. 平均通信成本: {avg_cost:.4f} bits')

if __name__ == "__main__":
    reference_camera_id = 1
    unknown_camera_id = 7
    N_threshold = 5

    # 动态获取可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available! Please check your CUDA setup.")
    
    print(f"检测到 {num_gpus} 张 GPU，准备分配任务...")

    # 获取所有需要处理的帧列表
    all_frames = list(range(frame_start, frame_end + 1, frame_step))
    
    # 将帧列表均匀切分给每张 GPU
    frames_split = np.array_split(all_frames, num_gpus)

    torch.multiprocessing.set_start_method('spawn', force=True)
    processes = []
    
    for gpu_id in range(num_gpus):
        frames_for_this_gpu = frames_split[gpu_id].tolist()
        if len(frames_for_this_gpu) == 0:
            continue
            
        p = torch.multiprocessing.Process(
            target=main_worker, 
            args=(reference_camera_id, unknown_camera_id, frames_for_this_gpu, N_threshold, gpu_id)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    print("所有 GPU 任务执行完毕！")
