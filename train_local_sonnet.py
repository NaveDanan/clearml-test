import time
import argparse
import os
import random
import logging
import numpy as np
import glob
import re
import platform
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from copy import copy
import yaml
import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from scipy.signal import butter, filtfilt
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
import clearml
from clearml import Task
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import psutil

# Initialize ClearML Task
# This should be one of the first lines in your script
task = Task.init(project_name='CNN-ResNet-Optimization', task_name='advanced_resnet_training', output_uri=True)

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Call set_seed early
set_seed(42)

def load_config(config_path=None):
    """Load configuration from YAML file"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded configuration from: {config_path}")
        return config
    elif os.path.exists('training_config.yaml'):
        with open('training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"✅ Loaded default configuration from: training_config.yaml")
        return config
    else:
        print("⚠️  No configuration file found, using command line arguments only")
        return None

def merge_config_with_args(config, args):
    """Merge YAML configuration with command line arguments"""
    if config is None:
        return args
    
    print("🔧 Merging YAML configuration with command line arguments...")
    
    # Command line arguments take precedence over config file
    # Only update args if they weren't explicitly set via command line
    
    # Model configuration
    if hasattr(args, 'model_name') and args.model_name == 'resnet152':  # Default value
        args.model_name = config.get('model', {}).get('name', args.model_name)
    
    # Data configuration
    data_config = config.get('data', {})
    if hasattr(args, 'image_size') and args.image_size == 224:  # Default value
        args.image_size = data_config.get('image_size', args.image_size)
    if hasattr(args, 'batch') and args.batch == 32:  # Default value
        args.batch = data_config.get('batch_size', args.batch)
    if hasattr(args, 'workers') and args.workers == 6:  # Default value
        args.workers = data_config.get('num_workers', args.workers)
    
    # Training configuration
    training_config = config.get('training', {})
    if hasattr(args, 'epochs') and args.epochs == 50:  # Default value
        args.epochs = training_config.get('epochs', args.epochs)
    if hasattr(args, 'lr') and args.lr == 0.001:  # Default value
        args.lr = training_config.get('learning_rate', args.lr)
    if hasattr(args, 'weight_decay') and args.weight_decay == 0.0001:  # Default value
        args.weight_decay = training_config.get('weight_decay', args.weight_decay)
    if hasattr(args, 'scheduler') and args.scheduler == 'cosine':  # Default value
        args.scheduler = training_config.get('scheduler', args.scheduler)
    if hasattr(args, 'patience') and args.patience == 10:  # Default value
        args.patience = training_config.get('patience', args.patience)
    
    # ClearML configuration
    clearml_config = config.get('clearml', {})
    if hasattr(args, 'stream_artifacts') and not args.stream_artifacts:  # Default False
        args.stream_artifacts = clearml_config.get('stream_artifacts', args.stream_artifacts)
    
    # Add config object to args for advanced settings
    args.config = config
    
    print("✅ Configuration merged successfully")
    return args

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, task=None, stream_artifacts=False, min_delta=0.001):
        """
        Enhanced Early Stopping with minimum delta requirement
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            task (Task): ClearML Task object for logging.
            stream_artifacts (bool): If True, uploads artifacts immediately.
            min_delta (float): Minimum change in monitored quantity to qualify as improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.task = task
        self.stream_artifacts = stream_artifacts
        self.min_delta = min_delta

    def __call__(self, val_loss, model, save_path, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path, epoch)
        elif val_loss >= self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path, epoch)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, save_path, epoch):
        """Save model when validation loss decreases significantly."""
        if self.verbose:
            print(f'Validation loss improved ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
        ckpt_path = os.path.join(save_path, 'checkpoint.pth')
        
        # Save comprehensive checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, ckpt_path)
        
        if self.stream_artifacts and self.task:
            self.task.upload_artifact('checkpoint', ckpt_path)
class AdvancedCustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, augment_transform=None, 
                 resize=None, mixup_alpha=0.2, cutmix_alpha=1.0):
        """
        Enhanced dataset with advanced augmentation techniques
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment_transform = augment_transform
        self.resize = resize
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.resize:
            image = self.resize(image)
            
        if self.augment_transform:
            image = self.augment_transform(image)
            
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
    def mixup_data(self, x, y, alpha=1.0):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
            
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

class MetricsTracker:
    def __init__(self, task=None):
        """Track comprehensive training metrics"""
        self.task = task
        self.metrics = defaultdict(list)
        self.best_metrics = {}
        
    def update(self, phase, epoch, **kwargs):
        """Update metrics for current epoch"""
        for key, value in kwargs.items():
            metric_name = f"{phase}_{key}"
            self.metrics[metric_name].append(value)
            
            # Log to ClearML
            if self.task:
                self.task.get_logger().report_scalar(
                    title=phase.capitalize(), 
                    series=key.capitalize(), 
                    value=value, 
                    iteration=epoch
                )
    
    def update_best(self, metric_name, value, epoch):
        """Track best metrics"""
        if metric_name not in self.best_metrics or value > self.best_metrics[metric_name]['value']:
            self.best_metrics[metric_name] = {'value': value, 'epoch': epoch}
    
    def get_summary(self):
        """Get training summary"""
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[key] = {
                    'final': values[-1],
                    'best': max(values) if 'acc' in key or 'f1' in key or 'auc' in key else min(values),
                    'mean': np.mean(values),
                    'std': np.std(values)
                }
        return summary

# Function to print CPU and memory usage
def print_system_usage():
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory_info.percent}%")

# Define a picklable transform class for multiprocessing
class MultiplyTransform:
    def __init__(self, multiplier):
        self.multiplier = multiplier
    
    def __call__(self, x):
        return [x] * self.multiplier

# Enhanced augmentation with state-of-the-art techniques
def get_advanced_augmentation(image_size, phase='train', config=None):
    """Get advanced augmentation pipeline based on training phase and config"""
    
    if phase == 'train':
        # Use config values if available, otherwise use defaults
        aug_config = config.get('augmentation', {}).get('train', {}) if config else {}
        
        transforms_list = [
            transforms.Resize((int(image_size * 1.143), int(image_size * 1.143))),
            transforms.RandomCrop((image_size, image_size)),
        ]
        
        # Conditional augmentations based on config
        if aug_config.get('horizontal_flip', 0.5) > 0:
            transforms_list.append(transforms.RandomHorizontalFlip(p=aug_config.get('horizontal_flip', 0.5)))
        
        if aug_config.get('vertical_flip', 0.1) > 0:
            transforms_list.append(transforms.RandomVerticalFlip(p=aug_config.get('vertical_flip', 0.1)))
        
        if aug_config.get('rotation_degrees', 15) > 0:
            transforms_list.append(transforms.RandomRotation(degrees=aug_config.get('rotation_degrees', 15)))
        
        # Color jitter
        color_jitter = aug_config.get('color_jitter', {
            'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1
        })
        if any(color_jitter.values()):
            transforms_list.append(transforms.ColorJitter(**color_jitter))
        
        transforms_list.append(transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5))
        
        # Gaussian blur
        if aug_config.get('gaussian_blur_prob', 0.3) > 0:
            transforms_list.append(transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=aug_config.get('gaussian_blur_prob', 0.3)))
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Random erasing (applied after ToTensor)
        if aug_config.get('random_erasing_prob', 0.3) > 0:
            transforms_list.append(transforms.RandomErasing(
                p=aug_config.get('random_erasing_prob', 0.3), 
                scale=(0.02, 0.33), 
                ratio=(0.3, 3.3)
            ))
        
        return transforms.Compose(transforms_list)
    else:
        # Validation/Test - only resize and normalize
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# Enhanced data preparation function
def advanced_data_prep(data_dir, workers, batch_size, image_size, use_weighted_sampling=True, config=None):
    """Enhanced data preparation with advanced techniques"""
    class_names = ['pass', 'fail']
    file_paths = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            raise ValueError(f"Class directory not found: {class_dir}")
            
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        for img_name in images:
            file_paths.append(os.path.join(class_dir, img_name))
            labels.append(label)

    print(f"Found {len(file_paths)} images across {len(class_names)} classes")
    print(f"Class distribution: {dict(zip(class_names, [labels.count(i) for i in range(len(class_names))]))}")

    # Stratified split for better class balance
    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=0.15, stratify=labels, random_state=42
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.176, stratify=train_labels, random_state=42
    )

    # Advanced augmentation with config support
    train_transform = get_advanced_augmentation(image_size, 'train', config)
    val_transform = get_advanced_augmentation(image_size, 'val', config)
    test_transform = get_advanced_augmentation(image_size, 'test', config)

    print("Preparing enhanced datasets...")
    print_system_usage()

    # Create datasets
    train_dataset = AdvancedCustomDataset(train_files, train_labels, transform=train_transform)
    val_dataset = AdvancedCustomDataset(val_files, val_labels, transform=val_transform)
    test_dataset = AdvancedCustomDataset(test_files, test_labels, transform=test_transform)

    # Enhanced weighted sampling if requested
    if use_weighted_sampling:
        class_counts = torch.tensor([train_labels.count(0), train_labels.count(1)], dtype=torch.float)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label].item() for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        shuffle_train = False
    else:
        sampler = None
        shuffle_train = True

    # Get hardware config
    hardware_config = config.get('hardware', {}) if config else {}
    pin_memory = hardware_config.get('pin_memory', torch.cuda.is_available())
    persistent_workers = hardware_config.get('persistent_workers', workers > 0)

    # Data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset, 
        sampler=sampler,
        batch_size=batch_size, 
        shuffle=shuffle_train,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True  # For stable batch norm
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    print(f"Enhanced data preparation completed!")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print_system_usage()

    return train_loader, val_loader, test_loader, class_weights.tolist() if use_weighted_sampling else None
################################
################################

###############################
# utils.general.py
###############################
# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))  # NumExpr max threads

def increment_path(path, exist_ok=False, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic

    if (path.exists() and exist_ok) or (not path.exists()):
        print(f'path in increment is: {path}')
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        print(f'path in increment is: {path}{sep}{n}')
        return f"{path}{sep}{n}"  # update path


def set_logging(rank=-1):
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    # init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else ''


def isdocker():
    # Is environment a Docker container
    return Path('/workspace').exists()  # or Path('/.dockerenv').exists()


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str
###############################
###############################

###############################
# resnet_metrics.py
###############################
def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

def ap_per_class(tp, conf, pred_cls, target_cls, v5_metric=False, plot=False, save_dir='.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j], v5_metric=v5_metric)
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        # The following functions are not defined in the provided script, commenting them out.
        # plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        # plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        # plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

def compute_ap(recall, precision, v5_metric=False):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
        v5_metric: Assume maximum recall to be 1.0, as in YOLOv5, MMDetetion etc.
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    if v5_metric:  # New YOLOv5 metric, same as MMDetection and Detectron2 repositories
        mrec = np.concatenate(([0.], recall, [1.0]))
    else:  # Old YOLOv5 metric, i.e. default YOLOv7 metric
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
    mpre = np.concatenate(([1.], precision, [0.]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc, nc)) # Adjusted for simple classification
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, preds, labels):
        """
        Process a batch of predictions and ground truth labels for classification class

        Arguments:
            preds (Tensor[N]), predicted class labels
            labels (Tensor[N]), ground truth class labels
        """
        if preds.dim() != 1 or labels.dim() != 1:
            raise ValueError("Labels and detections must be 1-dimensional tensors")
        # Ensure that the class indices are int
        preds = preds.int()
        labels = labels.int()

        for p, l in zip(preds, labels):
            if l < self.nc and p < self.nc:
                self.matrix[l, p] += 1


    def get_matrix(self):
        return self.matrix

    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix
            # Normalization can be added here if desired
            # array = self.matrix / (self.matrix.sum(0).reshape(1, self.nc) + 1E-6)  # normalize

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.nc < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.nc  # apply names to ticklabels

            sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.0f', square=True,
                       xticklabels=names if labels else "auto",
                       yticklabels=names if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('Predicted')
            fig.axes[0].set_ylabel('True')
            fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
            print(f"Saving confusion matrix in {Path(save_dir) / 'confusion_matrix.png'}")
            plt.close(fig)
        except Exception as e:
            print(f"Confusion matrix plot error: {e}")

    def print(self):
        for i in range(self.nc):
            print(' '.join(map(str, self.matrix[i])))
###############################
###############################

###############################
# utils.plots.py
###############################
# Settings
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only


def color_list():
    # Return first 10 plt colors as (r,g,b) https://stackoverflow.com/questions/51350872/python-from-color-name-to-rgb
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]  # or BASE_ (8), CSS4_ (148), XKCD_ (949)

def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap.mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close(fig)

def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])

def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    # https://stackoverflow.com/questions/28536191/how-to-filter-smooth-with-scipy-numpy
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)  # forward-backward filter

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color))  # plot
    if label:
        fontsize = max(round(max(img.size) / 40), 12)
        font = ImageFont.truetype("Arial.ttf", fontsize)
        txt_width, txt_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - txt_height + 4, box[0] + txt_width, box[1]], fill=tuple(color))
        draw.text((box[0], box[1] - txt_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)

def plot_wh_methods():  # from utils.plots import *; plot_wh_methods()
    # Compares the two methods for width-height anchor multiplication
    # https://github.com/ultralytics/yolov3/issues/168
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), tight_layout=True)
    plt.plot(x, ya, '.-', label='YOLOv3')
    plt.plot(x, yb ** 2, '.-', label='YOLOR ^2')
    plt.plot(x, yb ** 1.6, '.-', label='YOLOR ^1.6')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.legend()
    fig.savefig('comparison.png', dpi=200)
    plt.close(fig)

def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    # Plot image grid with labels

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalise
    if np.max(images[0]) <= 1:
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)  # font thickness
    bs, _, h, w = images.shape  # batch size, _, height, width
    bs = min(bs, max_subplots)  # limit plot images
    ns = np.ceil(bs ** 0.5)  # number of subplots (square)

    # Check if we should resize
    scale_factor = max_size / max(h, w)
    if scale_factor < 1:
        h = math.ceil(scale_factor * h)
        w = math.ceil(scale_factor * w)

    colors = color_list()  # list of colors
    mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
    for i, img in enumerate(images):
        if i == max_subplots:  # if last batch has fewer images than we expect
            break

        block_x = int(w * (i // ns))
        block_y = int(h * (i % ns))

        img = img.transpose(1, 2, 0)
        if scale_factor < 1:
            img = cv2.resize(img, (w, h))

        mosaic[block_y:block_y + h, block_x:block_x + w, :] = img

        # Draw image filename labels
        if paths:
            label = Path(paths[i]).name[:40]  # trim to 40 char
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0, tl / 3, [220, 220, 220], thickness=tf,
                        lineType=cv2.LINE_AA)

        # Image border
        cv2.rectangle(mosaic, (block_x, block_y), (block_x + w, block_y + h), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(h, w) / ns, 1.0)  # ratio to limit image size
        mosaic = cv2.resize(mosaic, (int(ns * w * r), int(ns * h * r)), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(fname, cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB))  # cv2 save
        Image.fromarray(mosaic).save(fname)  # PIL save
    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=''):
    # Plot LR simulating training for full epochs
    optimizer, scheduler = copy(optimizer), copy(scheduler)  # do not modify originals
    y = []
    for _ in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_targets_txt():  # from utils.plots import *; plot_targets_txt()
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)
    plt.close(fig)


def plot_study_txt(path='', x=None):  # from utils.plots import *; plot_study_txt()
    # Plot study.txt generated by test.py
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    # ax = ax.ravel()

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    # for f in [Path(path) / f'study_coco_{x}.txt' for x in ['yolor-p6', 'yolor-w6', 'yolor-e6', 'yolor-d6']]:
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        # for i in range(7):
        #     ax[i].plot(x, y[i], '.-', linewidth=2, markersize=8)
        #     ax[i].set_title(s[i])

        j = y[3].argmax() + 1
        ax2.plot(y[6, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', '').replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')

    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(str(Path(path).name) + '.png', dpi=300)
    plt.close(fig)
    plt.close(fig2)


def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):
    # plot dataset labels
    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()  # classes, boxes
    nc = int(c.max() + 1)  # number of classes
    colors = color_list()
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # seaborn correlogram
    sns.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')  # faster
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    ax[0].hist(c, bins=np.linspace(0, nc, nc + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sns.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sns.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5  # center
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors[int(cls) % 10])  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close(fig)

    # loggers
    for k, v in loggers.items() or {}:
        if k == 'wandb' and v:
            v.log({"Labels": [v.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.jpg')]}, commit=False)


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):  # from utils.plots import *; plot_evolution()
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = fitness(x)
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 7]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')
    plt.close()


def profile_idetection(start=0, stop=0, labels=(), save_dir=''):
    # Plot iDetection '*.txt' per-image logs. from utils.plots import *; profile_idetection()
    fig, ax = plt.subplots(2, 4, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for fi, f in enumerate(files):
        try:
            results = np.loadtxt(f, ndmin=2).T[:, 90:-30]  # clip first and last rows
            n = results.shape[1]  # number of rows
            x = np.arange(start, min(stop, n) if stop else n)
            results = results[:, x]
            t = (results[0] - results[0].min())  # set t0=0s
            results[0] = x
            for i, a in enumerate(ax):
                if i < len(results):
                    label = labels[fi] if len(labels) else f.stem.replace('frames_', '')
                    a.plot(t, results[i], marker='.', label=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('time (s)')
                    # if fi == len(files) - 1:
                    #     a.set_ylim(bottom=0)
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (f, e))

    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)
    plt.close(fig)

def plot_confusion_matrix(epoch_num, cm, output_dir)->None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Expected (True)")
    plt.title(f"Confusion Matrix for epoch {epoch_num}")
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_epoch_{epoch_num}.png'))
    plt.close()

def plot_results_overlay(start=0, stop=0):  # from utils.plots import *; plot_results_overlay()
    # Plot training 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']  # legends
    t = ['Box', 'Objectness', 'Classification', 'P-R', 'mAP-F1']  # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(1, 5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])
                # y_smooth = butter_lowpass_filtfilt(y)
                # ax[i].plot(x, np.gradient(y_smooth), marker='.', label=s[j])

            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None  # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)
        plt.close(fig)


def plot_results(result_file, save_dir, start=0, stop=0):
    # plot training 'result.txt'
    fig, ax = plt.subplots(2, 2, figsize=(24, 24), tight_layout=True)
    ax = ax.ravel()

    data = np.loadtxt(result_file, skiprows=1, ndmin=2) # Read from results.txt
    if data.shape[0] < 1:
        print("Not enough data in results.txt to plot.")
        return

    # data extraction from .txt
    epochs = data[:, 0]
    gpu_mem = data[:, 1]
    train_loss = data[:, 2]
    train_acc = data[:, 3]
    val_loss = data[:, 4]
    val_acc = data[:, 5]
    precision = data[:, 6]
    recall = data[:, 7]
    f1_score_val = data[:, 8]
    map50_95 = data[:, 9]

    #sub (1.1): Train & Validation Loss vs Epochs
    ax[0].plot(epochs, train_loss, label="Train Loss")
    ax[0].plot(epochs, val_loss, label="Validation Loss")
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss Over Epochs')
    ax[0].legend()
    #sub (1.2): Train & Validation Accuracy vs Epochs
    ax[1].plot(epochs, train_acc, label="Train Acc")
    ax[1].plot(epochs, val_acc, label="Validation Acc")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy Over Epochs')
    ax[1].legend()
    #sub (2.1) Recall vs Epochs
    ax[2].plot(epochs, recall, label="Recall")
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Recall')
    ax[2].set_title('Recall Over Epochs')
    ax[2].legend()
    #sub (2.2) Precision vs Epochs
    ax[3].plot(epochs, precision, label="Precision")
    ax[3].set_xlabel('Epoch')
    ax[3].set_ylabel('Precision')
    ax[3].set_title('Precision Over Epochs')
    ax[3].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)
    plt.close(fig)

    # PR_Plot
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'o-', label="P vs R")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Plot')
    plt.grid(True)
    plt.legend()
    plt.savefig(Path(save_dir) / 'PR_Plot.png')
    plt.close()

    #F1_Score Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, f1_score_val, label="F1_Score")
    plt.xlabel('Epochs')
    plt.ylabel('F1_Score')
    plt.title('F1_Score vs Epochs')
    plt.legend()
    plt.savefig(Path(save_dir) / 'F1_Score.png')
    plt.close()


def plot_skeleton_kpts(im, kpts, steps, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        if steps == 3:
            conf1 = kpts[(sk[0]-1)*steps+2]
            conf2 = kpts[(sk[1]-1)*steps+2]
            if conf1<0.5 or conf2<0.5:
                continue
        if pos1[0]%640 == 0 or pos1[1]%640==0 or pos1[0]<0 or pos1[1]<0:
            continue
        if pos2[0] % 640 == 0 or pos2[1] % 640 == 0 or pos2[0]<0 or pos2[1]<0:
            continue
        cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)

# Enhanced ResNet with better architecture
class EnhancedResNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_name='resnet152'):
        """Enhanced ResNet with improved architecture"""
        super(EnhancedResNet, self).__init__()
        
        # Load pretrained model
        if model_name == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        else:
            self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get feature dimension
        num_ftrs = self.backbone.fc.in_features
        
        # Replace final classifier with enhanced version
        self.backbone.fc = nn.Identity()  # Remove original classifier
        
        # Enhanced classifier with dropout and batch norm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights with Xavier initialization"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


# Enhanced training function with best practices
def enhanced_train_model(args, model, train_loader, val_loader, device='cuda'):
    """Enhanced training with CNN best practices"""
    
    save_dir, num_epochs, patience = Path(args.save_dir), args.epochs, args.patience
    
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)
    last = wdir / 'last.pth'
    best = wdir / 'best.pth'
    results_file = save_dir / 'results.txt'
    summary_file = save_dir / 'summary.txt'
    log_dir = save_dir / 'tensorboard_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Enhanced loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Enhanced optimizer with different learning rates for backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},  # Lower LR for pretrained backbone
        {'params': classifier_params, 'lr': args.lr}       # Higher LR for new classifier
    ], weight_decay=args.weight_decay, eps=1e-8)
    
    # Enhanced learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, 
                              steps_per_epoch=len(train_loader), 
                              epochs=num_epochs)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Mixed precision training
    scaler = GradScaler('cuda')
    
    # Metrics tracking
    metrics_tracker = MetricsTracker(task)
    
    # TensorBoard logging
    writer = SummaryWriter(log_dir=log_dir)
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=patience, 
        verbose=True,
        task=task,
        stream_artifacts=args.stream_artifacts,
        min_delta=0.001
    )
    
    # Training state
    best_val_acc = 0.0
    best_val_f1 = 0.0
    start_time = time.time()
    
    # Results file header
    with open(results_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc,Val_Precision,Val_Recall,Val_F1,Val_AUC,LR\n")
    
    print(f"Starting enhanced training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}')
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Mixed precision backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Statistics
            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
            train_total += labels.size(0)
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_corrects/train_total:.4f}',
                'LR': f'{current_lr:.6f}'
            })
            
            # Update scheduler if OneCycleLR
            if args.scheduler == 'onecycle':
                scheduler.step()
        
        # Calculate training metrics
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_corrects / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy()[:, 1])  # Probability of positive class
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_corrects / val_total
        
        # Advanced metrics
        val_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics tracker
        metrics_tracker.update('train', epoch, loss=epoch_train_loss, accuracy=epoch_train_acc)
        metrics_tracker.update('val', epoch, loss=epoch_val_loss, accuracy=epoch_val_acc, 
                             precision=val_precision, recall=val_recall, f1=val_f1, auc=val_auc)
        
        # Log to TensorBoard
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
        writer.add_scalar('Loss/Val', epoch_val_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_acc, epoch)
        writer.add_scalar('Accuracy/Val', epoch_val_acc, epoch)
        writer.add_scalar('F1/Val', val_f1, epoch)
        writer.add_scalar('AUC/Val', val_auc, epoch)
        
        # Save results
        with open(results_file, 'a') as f:
            f.write(f"{epoch},{epoch_train_loss:.6f},{epoch_train_acc:.6f},"
                   f"{epoch_val_loss:.6f},{epoch_val_acc:.6f},{val_precision:.6f},"
                   f"{val_recall:.6f},{val_f1:.6f},{val_auc:.6f},{current_lr:.8f}\n")
        
        # Print epoch results
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}')
        print(f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}')
        print(f'Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}')
        print(f'Learning Rate: {current_lr:.8f}')
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = epoch_val_acc
            
            # Save comprehensive checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': epoch_val_loss,
                'val_acc': epoch_val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'args': args
            }
            torch.save(checkpoint, best)
            
            if args.stream_artifacts:
                task.upload_artifact('best_model', best, metadata={'epoch': epoch, 'f1': val_f1})
        
        # Update scheduler
        if args.scheduler == 'plateau':
            scheduler.step(epoch_val_loss)
        elif args.scheduler == 'cosine':
            scheduler.step()
        
        # Early stopping check
        early_stopping(epoch_val_loss, model, args.save_dir, epoch)
        if early_stopping.early_stop:
            print(f'\nEarly stopping triggered at epoch {epoch+1}')
            break
    
    # Training completed
    training_time = time.time() - start_time
    print(f'\nTraining completed in {training_time//60:.0f}m {training_time%60:.0f}s')
    print(f'Best Val F1: {best_val_f1:.4f}, Best Val Acc: {best_val_acc:.4f}')
    
    # Save final checkpoint
    final_checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'training_time': training_time,
        'best_val_f1': best_val_f1,
        'best_val_acc': best_val_acc
    }
    torch.save(final_checkpoint, last)
    
    # Generate comprehensive summary
    metrics_summary = metrics_tracker.get_summary()
    
    with open(summary_file, 'w') as f:
        f.write(f"Enhanced CNN Training Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Training Time: {training_time/60:.2f} minutes\n")
        f.write(f"Epochs Completed: {epoch + 1}\n")
        f.write(f"Best Validation F1: {best_val_f1:.4f}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
        f.write(f"Final Learning Rate: {current_lr:.8f}\n")
        f.write(f"\nDetailed Metrics:\n")
        for metric, values in metrics_summary.items():
            f.write(f"{metric}: Final={values['final']:.4f}, Best={values['best']:.4f}\n")
    
    writer.close()
    
    # Upload artifacts if not streaming
    if not args.stream_artifacts:
        task.upload_artifact('best_model', best)
        task.upload_artifact('final_model', last)
        task.upload_artifact('training_summary', summary_file)
    
    return model, epoch + 1, summary_file

if __name__ == '__main__':
    # Set multiprocessing start method for Windows compatibility
    if platform.system() == 'Windows':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Enhanced CNN Training Script with Best Practices')
    
    # Data arguments
    parser.add_argument('-d', '--data-dir', type=str, default=os.path.abspath('data_dir'), 
                       help='Directory where the data is stored')
    parser.add_argument('--image-size', type=int, default=224, 
                       help='Input image size for training')
    parser.add_argument('-w', '--workers', type=int, default=6, 
                       help='Number of workers for data loading')
    parser.add_argument('-b', '--batch', type=int, default=32, 
                       help='Batch size for training')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='resnet152', 
                       choices=['resnet50', 'resnet101', 'resnet152'],
                       help='ResNet model variant to use')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('-e', '--epochs', type=int, default=50, 
                       help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001, 
                       help='Weight decay for regularization')
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['plateau', 'cosine', 'onecycle'],
                       help='Learning rate scheduler type')
    
    # Early stopping and checkpointing
    parser.add_argument('-p', '--patience', type=int, default=10, 
                       help='Patience for early stopping')
    parser.add_argument('--stream-artifacts', action='store_true',
                       help='Stream artifacts to ClearML during training')
    
    # Output arguments
    parser.add_argument('--project', type=str, default=os.path.abspath('runs/enhanced_train'), 
                       help='Project directory for saving results')
    parser.add_argument('-n', '--name', type=str, default='enhanced_resnet', 
                       help='Experiment name')
    parser.add_argument('--device', type=str, default='', 
                       help='Device to run on (auto-detect if empty)')
    
    # Advanced training options
    parser.add_argument('--use-weighted-sampling', action='store_true', default=True,
                       help='Use weighted sampling for class imbalance')
    parser.add_argument('--cross-validation', action='store_true',
                       help='Use cross-validation for robust evaluation')
    parser.add_argument('--test-time-augmentation', action='store_true',
                       help='Use test-time augmentation for final evaluation')
    
    # Configuration file support
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML configuration file (default: training_config.yaml)')
    
    # Parse arguments (handle unknown args for Jupyter compatibility)
    args, unknown = parser.parse_known_args()
    
    # === Load and merge configuration ===
    config = load_config(args.config)
    args = merge_config_with_args(config, args)
    
    # === ClearML Integration ===
    # Connect arguments to ClearML for hyperparameter tracking
    task.connect(args, name='Training_Configuration')
    
    # Log system information
    system_info = {
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    task.connect(system_info, name='System_Info')
    
    # Set device
    if not args.device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create save directory
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=False)
    print(f"Results will be saved to: {args.save_dir}")
    
    # === Data Loading ===
    print(f"Loading data from: {os.path.abspath(args.data_dir)}")
    
    # Connect to ClearML dataset
    try:
        dataset = clearml.Dataset.get(dataset_name="Dogs-vs-Cats", alias="wallak_data")
        data_dir = dataset.get_local_copy()
        print(f"Using ClearML dataset: {data_dir}")
    except Exception as e:
        print(f"ClearML dataset not found, using local data: {e}")
        data_dir = args.data_dir
    
    # Enhanced data preparation
    train_loader, val_loader, test_loader, class_weights = advanced_data_prep(
        data_dir=data_dir,
        workers=args.workers,
        batch_size=args.batch,
        image_size=args.image_size,
        use_weighted_sampling=args.use_weighted_sampling,
        config=config
    )
    
    print(f"Data loading completed:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    if class_weights:
        print(f"  Class weights: {class_weights}")
    
    # === Model Creation ===
    print(f"Creating enhanced {args.model_name} model...")
    model = EnhancedResNet(
        num_classes=2, 
        pretrained=args.pretrained, 
        model_name=args.model_name
    )
    
    # Load pretrained weights if available
    if os.path.exists("resnet152-b121ed2d.pth") and args.model_name == 'resnet152':
        print("Loading custom pretrained weights...")
        try:
            state_dict = torch.load("resnet152-b121ed2d.pth", weights_only=True)
            model.backbone.load_state_dict(state_dict, strict=False)
            print("Custom weights loaded successfully")
        except Exception as e:
            print(f"Could not load custom weights: {e}")
    
    model = model.to(device)
    
    # Log model architecture
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    model_info = {
        'model_name': args.model_name,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'pretrained': args.pretrained
    }
    task.connect(model_info, name='Model_Info')
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # === Training ===
    print("Starting enhanced training...")
    
    model, epochs_completed, summary_file = enhanced_train_model(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    print(f"Training completed after {epochs_completed} epochs")
    
    # === Final Evaluation ===
    print("Evaluating model on test set...")
    
    model.eval()
    test_corrects = 0
    test_total = 0
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if args.test_time_augmentation:
                # Test-time augmentation
                outputs_sum = torch.zeros(labels.size(0), 2).to(device)
                for _ in range(5):  # 5 augmented versions
                    with autocast(device_type='cuda'):
                        outputs = model(inputs)
                    outputs_sum += torch.softmax(outputs, dim=1)
                outputs = outputs_sum / 5
            else:
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    outputs = torch.softmax(outputs, dim=1)
            
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data).item()
            test_total += labels.size(0)
            
            all_test_preds.extend(preds.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
            all_test_probs.extend(outputs.cpu().numpy()[:, 1])
    
    # Calculate comprehensive test metrics
    test_acc = test_corrects / test_total
    test_precision = precision_score(all_test_labels, all_test_preds, average='binary', zero_division=0)
    test_recall = recall_score(all_test_labels, all_test_preds, average='binary', zero_division=0)
    test_f1 = f1_score(all_test_labels, all_test_preds, average='binary', zero_division=0)
    test_auc = roc_auc_score(all_test_labels, all_test_probs) if len(set(all_test_labels)) > 1 else 0.0
    
    # Log final test metrics to ClearML
    task.get_logger().report_scalar("Test_Metrics", "Accuracy", test_acc, 0)
    task.get_logger().report_scalar("Test_Metrics", "Precision", test_precision, 0)
    task.get_logger().report_scalar("Test_Metrics", "Recall", test_recall, 0)
    task.get_logger().report_scalar("Test_Metrics", "F1_Score", test_f1, 0)
    task.get_logger().report_scalar("Test_Metrics", "AUC", test_auc, 0)
    
    # Print final results
    print(f"\n{'='*50}")
    print(f"FINAL TEST RESULTS:")
    print(f"{'='*50}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"{'='*50}")
    
    # Save comprehensive summary
    final_summary = f"""
Enhanced CNN Training Summary
{'='*50}
Model: {args.model_name}
Dataset: {data_dir}
Training Configuration:
  - Epochs: {epochs_completed}
  - Batch Size: {args.batch}
  - Learning Rate: {args.lr}
  - Scheduler: {args.scheduler}
  - Image Size: {args.image_size}
  - Device: {device}

Final Test Metrics:
  - Accuracy: {test_acc:.4f}
  - Precision: {test_precision:.4f}
  - Recall: {test_recall:.4f}
  - F1-Score: {test_f1:.4f}
  - AUC: {test_auc:.4f}

Model Parameters:
  - Total: {total_params:,}
  - Trainable: {trainable_params:,}

Results saved to: {args.save_dir}
"""
    
    # Save to file
    with open(summary_file, 'a') as f:
        f.write(final_summary)
    
    # Log to ClearML console
    task.get_logger().report_text(final_summary, title="Final_Training_Summary")
    
    print(f"\nTraining completed successfully!")
    print(f"All results saved to: {args.save_dir}")
    print(f"Summary file: {summary_file}")
    
    # Close ClearML task
    task.close()