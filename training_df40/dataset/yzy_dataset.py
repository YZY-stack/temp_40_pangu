'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
'''

import sys
sys.path.append('.')
import os
import cv2
import yaml
import random
import torch
import numpy as np
from copy import deepcopy
import albumentations as A
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.sbi_api import SBI_API
from training.dataset.utils.bi_online_generation_yzy import random_get_hull
from training.dataset.SimSwap.test_one_image import self_blend

import warnings
warnings.filterwarnings('ignore')


class YZYDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label == 0]

        # Init SBI
        self.sbi = YZY_API(phase=mode,image_size=config['resolution'])


    def __getitem__(self, index):
        # Get the real image paths and labels
        real_image_path, real_label = self.real_imglist[index]
        fake_label = 1
        if not os.path.exists(real_image_path):
            real_image_path = real_image_path.replace('/Youtu_Pangu_Security_Public/', '/Youtu_Pangu_Security/public/')

        # Get the landmark paths for real images
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        landmark = self.load_landmark(real_landmark_path).astype(np.int32)

        # # Get the parsing mask
        # parsing_mask_path = real_image_path.replace('frames', 'parse_mask')
        # parsing_mask_path_wneck = real_image_path.replace('frames', 'parse_mask_wneck')
        # parsing_mask = cv2.imread(parsing_mask_path, cv2.IMREAD_GRAYSCALE)
        # parsing_mask_wneck = cv2.imread(parsing_mask_path_wneck, cv2.IMREAD_GRAYSCALE)
        # parising_mask_combine = (parsing_mask, parsing_mask_wneck)

        sri_path = real_image_path.replace('frames', 'sri_frames')
        parising_mask_combine = self.load_rgb(sri_path)

        # Load the real images
        real_image = self.load_rgb(real_image_path)
        real_image = np.array(real_image)  # Convert to numpy array

        # Randomly produce two mask types
        mask_regions = random.sample([1, 4, 5, 6], 2)
        mask1, mask2 = mask_regions

        # Generate the corresponding SBI sample
        rand_seed = random.randint(0, 666666)
        fake_image_1, real_image_1, mask_f1 = self.sbi(real_image.copy(), landmark.copy(), mask1, rand_seed, parising_mask_combine)
        fake_image_2, real_image_2, mask_f2 = self.sbi(real_image.copy(), landmark.copy(), mask2, rand_seed, parising_mask_combine)
        # zero_mask = np.zeros_like(mask_f1)
        
        # make sure some restrictions
        assert real_image_1.all() == real_image_2.all()
        
        # To tensor and normalize for fake and real images
        fake_image_trans_1 = self.normalize(self.to_tensor(fake_image_1))
        real_image_trans = self.normalize(self.to_tensor(real_image_1))
        fake_image_trans_2 = self.normalize(self.to_tensor(fake_image_2))

        return {
            "fake_1": (fake_image_trans_1, fake_label), 
            "fake_2": (fake_image_trans_2, fake_label), 
            "real": (real_image_trans, real_label)
        }

    def __len__(self):
        return len(self.real_imglist)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor and label tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors for fake and real data
        fake_images_1, fake_labels_1 = zip(*[data["fake_1"] for data in batch])
        fake_images_2, fake_labels_2 = zip(*[data["fake_2"] for data in batch])
        real_images, real_labels = zip(*[data["real"] for data in batch])

        # Stack the image, label, landmark, and mask tensors for fake and real data
        fake_images_1 = torch.stack(fake_images_1, dim=0)
        fake_images_2 = torch.stack(fake_images_2, dim=0)
        fake_images = torch.cat([fake_images_1, fake_images_2], dim=0)
        fake_labels = torch.LongTensor(fake_labels_1 + fake_labels_2)
        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)

        # Combine the fake and real tensors and create a dictionary of the tensors
        images = torch.cat([real_images, fake_images], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        
        data_dict = {
            'image': images,
            'label': labels,
            'landmark': None,
            'mask': None,
        }
        return data_dict


class YZY_API(SBI_API):
    def __init__(self,phase='train',image_size=256):
        super().__init__(phase=phase,image_size=image_size)
        

    def __call__(self,img,landmark=None,mask_region=None,rand_seed=None,parising_mask_combine=None):
        try:  
            assert landmark is not None, "landmark of the facial image should not be None."  
            if rand_seed is not None:
                random.seed(rand_seed)
            
            img_r,img_f,mask_f=self.self_blending(img,landmark,mask_region,rand_seed,parising_mask_combine)
            
            if self.phase=='train':  
                transformed=self.transforms(image=img_f.astype('uint8'),image1=img_r.astype('uint8'))  
                img_f=transformed['image']  
                img_r=transformed['image1']  
            return img_f,img_r,mask_f  
        except Exception as e:  
            print(e)  
            return None,None,None
        
    
    def self_blending(self,img,landmark,mask_region,rand_seed=None,parising_mask_combine=None):
        if rand_seed is not None:
            np.random.seed(rand_seed)  # set the seed for np.random
            
        H,W=len(img),len(img[0])
        if np.random.rand()<0.25:
            landmark=landmark[:68]
        
        if mask_region == 4:
            mask = parising_mask_combine[0]
        elif mask_region == 5:
            mask = parising_mask_combine[1]
        else:
            mask=random_get_hull(landmark,img,mask_region)[0][:,:,0]

        source = img.copy()
        if np.random.rand()<0.5:
            source = self.source_transforms(image=source.astype(np.uint8))['image']
        else:
            img = self.source_transforms(image=img.astype(np.uint8))['image']

        source, mask = self.randaffine(source,mask)

        img_blended,mask=self.dynamic_blend(source,img,mask,rand_seed)
        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img,img_blended,mask
    

    def dynamic_blend(self,source,target,mask,rand_seed=None):
        if rand_seed is not None:
            np.random.seed(rand_seed)  # set the seed for np.random
        mask_blured = self.get_blend_mask(mask, rand_seed)
        blend_list=[0.25,0.5,0.75,1,1,1]
        blend_ratio = blend_list[np.random.randint(len(blend_list))]
        mask_blured*=blend_ratio
        img_blended=(mask_blured * source + (1 - mask_blured) * target)
        return img_blended,mask_blured

    
    def get_blend_mask(self, mask, rand_seed=None):
        if rand_seed is not None:
            np.random.seed(rand_seed)  # set the seed for np.random
        H,W=mask.shape
        size_h=np.random.randint(192,257)
        size_w=np.random.randint(192,257)
        mask=cv2.resize(mask,(size_w,size_h))
        kernel_1=random.randrange(5,26,2)
        kernel_1=(kernel_1,kernel_1)
        kernel_2=random.randrange(5,26,2)
        kernel_2=(kernel_2,kernel_2)
        
        mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
        mask_blured = mask_blured/(mask_blured.max())
        mask_blured[mask_blured<1]=0
        
        mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5,46))
        mask_blured = mask_blured/(mask_blured.max())
        mask_blured = cv2.resize(mask_blured,(W,H))
        return mask_blured.reshape((mask_blured.shape+(1,)))



def create_bbox_mask(image, landmarks, margin=0):
    # Convert landmarks to a NumPy array if not already
    landmarks = np.array(landmarks)

    # Find the minimum and maximum x and y coordinates
    min_x, min_y = np.min(landmarks, axis=0)
    max_x, max_y = np.max(landmarks, axis=0)

    # Calculate width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y

    # Find the maximum of the two to get the side length of the square
    max_side = max(width, height)

    # Adjust the bounding box to be a square
    min_x = min_x - ((max_side - width) / 2)
    max_x = min_x + max_side
    min_y = min_y - ((max_side - height) / 2)
    max_y = min_y + max_side

    # Add margin
    min_x = max(0, min_x - margin)
    min_y = max(0, min_y - margin)
    max_x = min(image.shape[1], max_x + margin)
    max_y = min(image.shape[0], max_y + margin)

    # Convert coordinates to integers
    min_x, min_y, max_x, max_y = map(int, [min_x, min_y, max_x, max_y])

    # Create a black image of the same size as the input image
    mask = np.zeros_like(image[:, :, 0])

    # Set the pixel values within the bounding box to 255
    mask[min_y:max_y, min_x:max_x] = 255

    return mask






if __name__ == '__main__':
    with open('/data/home/zhiyuanyan/DeepfakeBenchv2/training/config/detector/sbi.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('./training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config2['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = '/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/DeepfakeBenchv2/preprocessing/dataset_json'
    config.update(config2)
    train_set = YZYDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break