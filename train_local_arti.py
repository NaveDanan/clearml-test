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
from torch.amp import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use non-interactive backend
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import clearml
from clearml import Task

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import psutil

print("Searching for resnet152 model...")
if os.path.exists('resnet152-b121ed2d.pth'):
    print("resnet152 model found.")
else:
    print("Downloading resnet152 model...")
    torch.hub.download_url_to_file('https://download.pytorch.org/models/resnet152-b121ed2d.pth', 'resnet152-b121ed2d.pth')

# Initialize ClearML Task
# This should be one of the first lines in your script
task = Task.init(project_name='new-ResNet-pytorch', task_name='train_local_arti', output_uri=True)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, task=None, stream_artifacts=False):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            task (Task): ClearML Task object for logging.
            stream_artifacts (bool): If True, uploads artifacts immediately; otherwise uploads at the end.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.task = task
        self.stream_artifacts = stream_artifacts

    def __call__(self,val_loss, model, save_path):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, save_path):
        ''' save model when validation loss decreases.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model ...')
        ckpt_path = os.path.join(save_path, 'checkpoint.pth')
        torch.save(model.state_dict(), ckpt_path)
        if self.stream_artifacts:
            self.task.upload_artifact('checkpoint', ckpt_path)
#######################
# Custom dataset class
#######################
class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, augment_transform=None, resize=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment_transform = augment_transform
        self.resize = resize

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.resize:
            image = self.resize(image)
        if self.augment_transform:
            # The augment_transform returns a list of augmented images
            images = self.augment_transform(image)
            image = images[0]  # Select the first augmented image for simplicity
        if self.transform:
            image = self.transform(image)
        return image, label

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

# Define augment_transform to handle PIL images
def get_augment_transform(multiplier, aug_params):
    # This function uses the connected augmentation parameters
    transforms_list = []
    
    if aug_params.get('RandomHorizontalFlip'):
        transforms_list.append(transforms.RandomHorizontalFlip())
    
    if aug_params.get('RandomRotation', 0) > 0:
        transforms_list.append(transforms.RandomRotation(aug_params.get('RandomRotation', 0)))
    
    if aug_params.get('ColorJitter'):
        transforms_list.append(transforms.ColorJitter(**aug_params.get('ColorJitter', {})))
    
    if aug_params.get('RandomAffine'):
        transforms_list.append(transforms.RandomAffine(**aug_params.get('RandomAffine', {})))
    
    transforms_list.append(MultiplyTransform(multiplier))
    
    return transforms.Compose(transforms_list)


# The main data preparation function
def data_prep(data_dir, multiplier, workers, batch_size, image_size, aug_params):
    class_names = ['pass', 'fail']
    file_paths = []
    labels = []

    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            file_paths.append(os.path.join(class_dir, img_name))
            labels.append(label)

    train_files, test_files, train_labels, test_labels = train_test_split(file_paths, labels, test_size=0.15, stratify=labels)
    train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.176, stratify=train_labels)

    resize_transform = transforms.Resize((int(image_size * 1.333), image_size))
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    augment_transform = get_augment_transform(multiplier, aug_params)

    print("Preparing training dataset...")
    print_system_usage()  # Print initial system usage

    # Create datasets
    train_dataset = CustomDataset(train_files, train_labels, transform=base_transform, augment_transform=augment_transform, resize=resize_transform)
    val_dataset = CustomDataset(val_files, val_labels, transform=base_transform, resize=resize_transform)
    test_dataset = CustomDataset(test_files, test_labels, transform=base_transform, resize=resize_transform)

    class_counts = torch.tensor([train_labels.count(0), train_labels.count(1)], dtype=torch.float)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label].item() for label in train_labels]

    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    print(f"Data preparation completed!")
    print_system_usage()  # Print final system usage

    return train_loader, val_loader, test_loader
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

###############################
# Model training
###############################
def train_model(args, scheduler, model, train_loader, val_loader, criterion, optimizer, device='cpu'):

    save_dir, num_epochs, patience = Path(args.save_dir), args.epochs, args.patience
    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True) #make dir
    last = wdir / 'last.pth'
    best = wdir / 'best.pth'
    results_file = save_dir / 'results.txt'
    summery = save_dir / 'summery.txt'
    log_dir = save_dir / 'log_dir'
    log_dir.mkdir(parents=True, exist_ok=True) #make dir
    cm_dir = save_dir / 'Confusion_Matrix_plots'
    cm_dir.mkdir(parents=True, exist_ok=True)
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_acc_epoch = 0

    # ClearML automatically captures TensorBoard logs.
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging directory: {log_dir}")

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    scaler = GradScaler('cuda') # # Initialize GradScaler for mixed precision training

    with open(results_file, 'w') as f:
        f.write("Epoch GPU_mem Train_Loss Train_Acc Val_Loss Val_Acc Precision Recall F1_Score mAP@0.5:0.95\n")

    # Initialize confusion matrix
    Confusion_Matrix = ConfusionMatrix(nc=2)  # Assuming binary classification: pass, fail
    early_stopping = EarlyStopping(
        patience=patience, 
        verbose=True,
        task=task,
        stream_artifacts=args.stream_artifacts
    )
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        mloss = torch.zeros(4, device=device)  # mean losses
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'm-loss', 'obj', 'cls', 'total', 'batch', 'img_size'))

        optimizer.zero_grad()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()  # Set model to evaluation mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            # Iterate over data
            for i, (inputs, labels) in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # No gradients in evaluation phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

                # Logging for the progress bar
                if phase == 'train':
                    mloss = (mloss * i + torch.tensor([loss.item(), 0, 0, 0], device=device)) / (i + 1)  # Update mean losses
                    mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    s = ('%10s' * 2 + '%10.4g' * 6) % (
                        '%g/%g' % (epoch, num_epochs - 1), mem, *mloss, labels.size(0), inputs.size(-1))
                    pbar.set_description(s)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects / len(dataloader.dataset)

            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            # Update confusion matrix
            if phase == 'val':
                Confusion_Matrix.process_batch(all_preds, all_labels)

            print(f'\n{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                writer.add_scalar('Loss/train', epoch_loss, epoch)
                writer.add_scalar('Accuracy/train', epoch_acc, epoch)
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                writer.add_scalar('Loss/val', epoch_loss, epoch)
                writer.add_scalar('Accuracy/val', epoch_acc, epoch)

                # Deep copy the model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, best)
                    if args.stream_artifacts:
                        task.upload_artifact(name='best_model', artifact_object=best, metadata={'epoch': epoch})
                    best_acc_epoch = epoch

        # This block should be outside the phase loop, running once per epoch
        precision = precision_score(all_labels.numpy(), all_preds.numpy(), average='binary', zero_division=0)
        recall = recall_score(all_labels.numpy(), all_preds.numpy(), average='binary', zero_division=0)
        f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average='binary', zero_division=0)

        # For classification, mAP is not standard. We log precision as a proxy.
        mAP50 = precision
        mAP50_95 = precision

        # cm = confusion_matrix(all_labels.numpy(), all_preds.numpy())
        # print(f'confusion matrix:\n {cm}')

        # Log results to result.txt
        with open(results_file, 'a') as f:
            f.write(f"{epoch} {mem[:-1]} {train_loss_history[-1]:.4f} {train_acc_history[-1]:.4f} {val_loss_history[-1]:.4f} {val_acc_history[-1]:.4f} {precision:.4f} {recall:.4f} {f1:.4f} {mAP50_95:.4f}\n")

        # # Saving confusion matrix plot for the current epoch
        # plot_confusion_matrix(output_dir=cm_dir, epoch_num=epoch, cm=cm)

        # Step the scheduler
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss_history[-1])
        else:
            scheduler.step()

        print()

        # Early stopping check
        early_stopping(val_loss_history[-1], model, args.save_dir)
        if early_stopping.early_stop:
            print(f'Training stopped early after {epoch + 1} epochs due to no improvement in validation loss.')
            break

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} at epoch {best_acc_epoch}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Close the TensorBoard writer
    writer.close()

    # Save performance graphs from the results file
    plot_results(results_file, save_dir=save_dir)

    # torch.save(model.state_dict(), last)
    # Save final confusion matrix plot
    final_cm = Confusion_Matrix.get_matrix()
    Confusion_Matrix.plot(save_dir=save_dir, names=['pass', 'fail'])

    # Final metrics calculation using the entire validation set's confusion matrix
    final_precision = final_cm[1,1] / (final_cm[1,1] + final_cm[0,1]) if (final_cm[1,1] + final_cm[0,1]) > 0 else 0
    final_recall = final_cm[1,1] / (final_cm[1,1] + final_cm[1,0]) if (final_cm[1,1] + final_cm[1,0]) > 0 else 0
    final_f1 = 2 * (final_precision * final_recall) / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

    print(f"Final Validation Precision: {final_precision}")
    print(f"Final Validation Recall: {final_recall}")
    print(f"Final Validation F1 Score: {final_f1}")

    with open(summery, 'w') as f:
        f.write(f"Run in {save_dir} summary Results:\n")
        f.write(f"Run-Time: {time_elapsed/60:.2f} [min]\n")
        f.write(f"Stopped after {epoch + 1} epochs\n")
        f.write(f"AVG GPU Usage: {mem}\n")
        f.write(f"Final Validation Precision: {final_precision:.4f}\n")
        f.write(f"Final Validation Recall: {final_recall:.4f}\n")
        f.write(f"Final Validation F1 Score: {final_f1:.4f}\n")
    if not args.stream_artifacts:
        # Upload the last and best model files to ClearML
        ckpt = wdir / 'checkpoint.pth'
        bst = wdir / 'best.pth'
        task.upload_artifact(name='checkpoint', artifact_object=ckpt, metadata={'Epoch': epoch})
        task.upload_artifact(name='best', artifact_object=bst, metadata={'Acc': best_acc_epoch})
    return model, epoch + 1, summery

if __name__ == '__main__':
    # Set multiprocessing start method for Windows compatibility
    if platform.system() == 'Windows':
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('-d', '--data-dir', type=str, default=os.path.abspath('data_dir'), help='Directory where the data is stored')
    parser.add_argument('--image-size', type=int, default=224, help='Choose image size to resize to train the model')
    parser.add_argument('-m', '--multi', type=int, default=6, help='Multiplier for dataset augmentation')
    parser.add_argument('-w', '--workers', type=int, default=6, help='Number of workers for data loading')
    parser.add_argument('-b', '--batch', type=int, default=32, help='Batch size for data loading')
    parser.add_argument('-o', '--output', type=str, default='.', help='Directory to save the best model')
    parser.add_argument('-n', '--name', type=str, default='clear', help='Name of the best model file')
    parser.add_argument('--project', type=str, default=os.path.abspath('runs/train'), help='Save to project/name')
    parser.add_argument('-e', '--epochs', type=int, default=3, help='Number of epochs for training')
    parser.add_argument('-p', '--patience', type=int, default=1, help='Patience for early stopping')
    parser.add_argument('--device', type=str, default='', help='Device to run the model on')
    # parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    # parser.add_argument('--weights', type=str, default='resnet152-b121ed2d.pth', help='Path to the pretrained weights file')
    parser.add_argument('--stream-artifacts', action='store_true', dest='stream_artifacts',
                         help='If set, upload checkpoint.pth and best.pth as soon as they are created; otherwise upload both only once after training finishes')
    # --- MODIFICATION ---
    # Use parse_known_args() to ignore extraneous arguments from Jupyter/Colab
    args, unknown = parser.parse_known_args()

    # --- ClearML Integration: Connect arguments and other parameters ---
    # `task` is already initialized at the top of the script
    task.connect(args, name='Hyperparameters')

    augmentation_params = {
        'RandomHorizontalFlip': True,
        'RandomRotation': 10,
        'ColorJitter': {'brightness': 0.1, 'contrast': 0.1, 'saturation': 0.1},
        'RandomAffine': {'degrees': (10, 40), 'translate': (0.1, 0.3), 'scale': (0.9, 1.1), 'shear': 5},
    }
    task.connect(augmentation_params, name='Augmentation')
    # -----------------------------------------------------------------

    print(f'\n\n The data directory is: {os.path.abspath(args.data_dir)}\n')
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=False)

    # --- ClearML Dataset Integration ---
    # This automatically logs which dataset version is being used
    The_data_dir = clearml.Dataset.get(dataset_name="Dogs-vs-Cats", alias="wallak_data").get_local_copy()

    # Data preparation
    train_loader, val_loader, test_loader = data_prep(The_data_dir,
                                                      multiplier=args.multi,
                                                      workers=args.workers,
                                                      batch_size=args.batch,
                                                      image_size=args.image_size,
                                                      aug_params=augmentation_params)

    # Load the pretrained model
    state_dict = torch.load(os.path.abspath("resnet152-b121ed2d.pth"), weights_only=False)

    # Create the model and load the state into your model
    model = models.resnet152()
    model.load_state_dict(state_dict, strict=False)

    # Check if GPU is available and move the model to the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Modify the final layer to match the number of classes in your dataset
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Assuming binary classification

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer_params = {'lr': 0.0001, 'momentum': 0.937, 'weight_decay': 0.0005}
    optimizer = optim.SGD(model.parameters(), **optimizer_params)

    # Learning rate scheduler
    scheduler_params = {'mode': 'min', 'factor': 0.01, 'patience': 3}
    scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)

    # --- ClearML Integration: Connect optimizer and scheduler parameters ---
    task.connect(optimizer_params, name='Optimizer')
    task.connect(scheduler_params, name='Scheduler')
    # ---------------------------------------------------------------------

    # Train the model
    model, epochs_completed, summery = train_model(model=model, scheduler=scheduler, train_loader=train_loader,
                                          val_loader=val_loader, criterion=criterion,
                                          optimizer=optimizer, args=args, device=device)


    # # Save the trained model
    # torch.save(model.state_dict(), os.path.join(args.save_dir, 'last_model.pth'))

    print(f'Training completed in {epochs_completed} epochs.')

    # Evaluate the model on the test set
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f'Test Acc: {test_acc:.4f}')

    # --- ClearML Integration: Log final metric and summary ---
    task.get_logger().report_scalar(title="Test", series="Accuracy", value=test_acc.item(), iteration=epochs_completed)

    summary_text = f'Test Acc: {test_acc:.4f}\n\nRun Configuration Brief:\n--------------------------\nAugmentation Multi: {args.multi}\nNumber of Workers: {args.workers}\nBatch Size: {args.batch}\nImage Size: {args.image_size}\nDevice: {device}\n'
    with open(summery, 'a') as f:
        f.write(summary_text)

    # Report the full summary text to ClearML console
    with open(summery, 'r') as f:
        full_summary = f.read()
    task.get_logger().report_text(full_summary, title="Run Summary", print_console=False)
    # ---------------------------------------------------------

    print(f"Script finished. All artifacts and results are saved in {args.save_dir} and uploaded to ClearML.")
    task.close()