import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from datetime import datetime
from normalize_staining import normalizeStaining
from datetime import datetime

# user input ------------------------------------------------------------------
BATCH_SIZE = 2
DATASET_ROOT = 'D:\\xpetrov\\ICIAR2018_BACH_Challenge\\Photos'
PARENT_DIR = 'D:\\xpetrov\\ICIAR2018_BACH_Challenge\\centralized'

dataset_definition = [
    # clietn 1
    {'Benign': 80, 'InSitu': 80, 'Invasive': 80, 'Normal': 80},# train
    {'Benign': 20, 'InSitu': 20, 'Invasive': 20, 'Normal': 20},# valid
    {'Benign': 0, 'InSitu': 0, 'Invasive': 0, 'Normal': 0},# test
    # client 2
    #{'Benign': 20, 'InSitu': 20, 'Invasive': 20, 'Normal': 20},# train
    #{'Benign': 6, 'InSitu': 6, 'Invasive': 6, 'Normal': 6},# valid
    #{'Benign': 4, 'InSitu': 4, 'Invasive': 4, 'Normal': 4},# test
    # client 3
    #{'Benign': 20, 'InSitu': 20, 'Invasive': 20, 'Normal': 20},# train
    #{'Benign': 6, 'InSitu': 6, 'Invasive': 6, 'Normal': 6},# valid
    #{'Benign': 4, 'InSitu': 4, 'Invasive': 4, 'Normal': 4},# test
    # server
    #{'Benign': 0, 'InSitu': 0, 'Invasive': 0, 'Normal': 0},# train
    #{'Benign': 6, 'InSitu': 6, 'Invasive': 6, 'Normal': 6},# valid
    #{'Benign': 4, 'InSitu': 4, 'Invasive': 4, 'Normal': 4},# test
]
# -----------------------------------------------------------------------------
# const values, do not change!
DATASET_SIZE = 400
CLASSES = ('Benign','InSitu','Invasive','Normal')
TRAIN_DIR = 'train'
VALID_DIR = 'valid'
TEST_DIR = 'test'

silo_ptr = dataset_definition.copy()

class_abbr = {
    'Benign': 'b',
    'InSitu': 'is',
    'Invasive': 'iv',
    'Normal': 'n'
}
# -----------------------------------------------------------------------------


class HE_Normalize(object):
    """Normalize staining appearence of H&E stained images"""
    def __call__(self, sample):
        norm_img, _, _ = normalizeStaining(np.array(sample))
        return norm_img


class CropPatches(object):
    """Crop the given image into square crops of size (kernel_size x kernel_size)"""
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, sample):
        kc, kh, kw = 3, self.kernel_size, self.kernel_size
        dc, dh, dw = 3, self.stride, self.stride
        patches = sample.unfold(0, kc, dc).unfold(1, kh, dh).unfold(2, kw, dw)
        patches = patches.contiguous().view(-1, kc, kh, kw)
        return patches

rotater = transforms.RandomRotation(
    degrees=5,
    interpolation=transforms.InterpolationMode.BILINEAR)

transform = transforms.Compose([
    HE_Normalize(),
    transforms.ToTensor(),
    transforms.Resize((256,256)),
    CropPatches(kernel_size=224, stride=8),
    transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
])


def check_dataset_definition():
    assert len(dataset_definition) % 3 == 0

    for i in range(len(dataset_definition)):
        for label in dataset_definition[i].keys():
            if label not in CLASSES:
                raise RuntimeError(
                    f"Unknown label '{label}' in dataset definition")    
    for label in CLASSES:
        images_of_the_label = 0
        for i in range(len(dataset_definition)):
            images_of_the_label += dataset_definition[i][label]
        if (images_of_the_label != 100):
            raise RuntimeError(
                f"The number of total '{label}' images should be 100")


def make_dirs():
    for i in range(len(dataset_definition) // 3):
        site_dir = 'site' + str(i+1)

        for label in dataset_definition[i].keys():
            train_path = os.path.join(PARENT_DIR, site_dir, TRAIN_DIR, label)
            valid_path = os.path.join(PARENT_DIR, site_dir, VALID_DIR, label)
            test_path = os.path.join(PARENT_DIR, site_dir, TEST_DIR, label)
            os.makedirs(train_path)
            os.makedirs(valid_path)
            os.makedirs(test_path)


def where_to_write(label: str, last_crop_of_image: bool):
    for i in range(len(silo_ptr)):
        if (silo_ptr[i][label] == 0):
            continue
        if (last_crop_of_image):
            silo_ptr[i][label] -= 1
        train_or_valid_or_test_dir =\
            'train' if i%3==0 else 'valid' if i%3==1 else 'test'
        silo_dir = 'site' + str(i//3+1)
        return silo_dir, train_or_valid_or_test_dir
    raise RuntimeError('Ran out of available capacity')


def log_progress(j):
    total = DATASET_SIZE / BATCH_SIZE
    print(f"{j}/{total}")

def log(msg: str):
    print(datetime.now(), msg)

def main():
    log('Validating federated dataset definition..')
    check_dataset_definition()
    log('Creating folders for federated dataset..')
    make_dirs()

    log('Loading dataset..')
    dataset = torchvision.datasets.ImageFolder(
        root=DATASET_ROOT,
        transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False)

    for j, data in enumerate(data_loader, 0):
        x, y = data
        bs, ncrops, c, h, w = x.size()
        x = torch.reshape(x, (-1, 3, 224, 224))
        y = y.view(-1,1).repeat(1,ncrops).view(-1)

        for i in range(x.size(0)):
            img_id = j * bs + i // ncrops + 1
            crop_id = i % ncrops + 1
            label = CLASSES[int(y[i])]
            silo_dir, train_or_valid_or_test_dir = where_to_write(label, i%ncrops==ncrops-1)
            image_name = PARENT_DIR + '/' + \
                silo_dir + '/' + \
                train_or_valid_or_test_dir + '/' + \
                label + '/' + \
                class_abbr[label] + str(img_id) + '_' + str(crop_id) + '.png'
            torchvision.utils.save_image(x[i], image_name)
        log_progress(j+1)
    log('Done!')


if __name__=='__main__':
    main()
