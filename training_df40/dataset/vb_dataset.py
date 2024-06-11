import os
import sys
sys.path.append('.')

import cv2
import yaml
import random
import torch
from collections import OrderedDict
import numpy as np
import numpy.linalg as npla
from copy import deepcopy
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
from dataset.albu import IsotropicResize
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from dataset.sbi_api import SBI_API



def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):
    if len(lmrks) != 68:
        raise Exception('works only with 68 landmarks')
    lmrks = np.array( lmrks.copy(), dtype=np.int32 )

    # #nose
    ml_pnt = (lmrks[36] + lmrks[0]) // 2
    mr_pnt = (lmrks[16] + lmrks[45]) // 2

    # mid points between the mid points and eye
    ql_pnt = (lmrks[36] + ml_pnt) // 2
    qr_pnt = (lmrks[45] + mr_pnt) // 2

    # Top of the eye arrays
    bot_l = np.array((ql_pnt, lmrks[36], lmrks[37], lmrks[38], lmrks[39]))
    bot_r = np.array((lmrks[42], lmrks[43], lmrks[44], lmrks[45], qr_pnt))

    # Eyebrow arrays
    top_l = lmrks[17:22]
    top_r = lmrks[22:27]

    # Adjust eyebrow arrays
    lmrks[17:22] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[22:27] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks

def get_image_hull_mask(image_shape, image_landmarks, eyebrows_expand_mod=1.0 ):
    hull_mask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)

    lmrks = expand_eyebrows(image_landmarks, eyebrows_expand_mod)

    r_jaw = (lmrks[0:9], lmrks[17:18])
    l_jaw = (lmrks[8:17], lmrks[26:27])
    r_cheek = (lmrks[17:20], lmrks[8:9])
    l_cheek = (lmrks[24:27], lmrks[8:9])
    nose_ridge = (lmrks[19:25], lmrks[8:9],)
    r_eye = (lmrks[17:22], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    l_eye = (lmrks[22:27], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    nose = (lmrks[27:31], lmrks[31:36])
    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

    for item in parts:
        merged = np.concatenate(item)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(merged), (1,) )

    return hull_mask



class VideoBlendAPI(SBI_API):
    def __init__(self, phase='train', image_size=224):
        super().__init__(phase, image_size)

    def __call__(self, img, landmark=None):
        try:
            assert landmark is not None, "landmark of the facial image should not be None."
            img_r, img_f, mask_f = self.self_blending(img.copy(), landmark.copy())
            return img_f, img_r, mask_f
        except Exception as e:
            print(e)
            return None, None, None

    
    def self_blending(self, img, landmark):
        h, w = len(img), len(img[0])
        image_shape = img.shape
        hull = get_image_hull_mask(image_shape, landmark)

        def process(w,h, data ):
            d = {}
            cur_lc = 0
            all_lines = []
            for s, pts_loop_ar in data:
                lines = []
                for pts, loop in pts_loop_ar:
                    pts_len = len(pts)
                    lines.append ( [ [ pts[i], pts[(i+1) % pts_len ] ]  for i in range(pts_len - (0 if loop else 1) ) ] )
                lines = np.concatenate (lines)

                lc = lines.shape[0]
                all_lines.append(lines)
                d[s] = cur_lc, cur_lc+lc
                cur_lc += lc
            all_lines = np.concatenate (all_lines, 0)

            #calculate signed distance for all points and lines
            line_count = all_lines.shape[0]
            pts_count = w*h

            all_lines = np.repeat ( all_lines[None,...], pts_count, axis=0 ).reshape ( (pts_count*line_count,2,2) )

            pts = np.empty( (h,w,line_count,2), dtype=np.float32 )
            pts[...,1] = np.arange(h)[:,None,None]
            pts[...,0] = np.arange(w)[:,None]
            pts = pts.reshape ( (h*w*line_count, -1) )

            a = all_lines[:,0,:]
            b = all_lines[:,1,:]
            pa = pts-a
            ba = b-a
            ph = np.clip ( np.einsum('ij,ij->i', pa, ba) / np.einsum('ij,ij->i', ba, ba), 0, 1 )
            dists = npla.norm ( pa - ba*ph[...,None], axis=1).reshape ( (h,w,line_count) )

            def get_dists(name, thickness=0):
                s,e = d[name]
                result = dists[...,s:e]
                if thickness != 0:
                    result = np.abs(result)-thickness
                return np.min (result, axis=-1)

            return get_dists

        l_eye = landmark[42:48]
        r_eye = landmark[36:42]
        l_brow = landmark[22:27]
        r_brow = landmark[17:22]
        mouth = landmark[48:60]

        up_nose = np.concatenate( (landmark[27:31], landmark[33:34]) )
        down_nose = landmark[31:36]
        nose = np.concatenate ( (up_nose, down_nose) )

        gdf = process ( w,h,
                            (
                            ('eyes',  ((l_eye, True), (r_eye, True)) ),
                            ('brows', ((l_brow, False), (r_brow,False)) ),
                            ('up_nose', ((up_nose, False),) ),
                            ('down_nose', ((down_nose, False),) ),
                            ('mouth', ((mouth, True),) ),
                            )
                            )

        eyes_fall_dist = w // 32
        eyes_thickness = max( w // 64, 1 )

        brows_fall_dist = w // 32
        brows_thickness = max( w // 256, 1 )

        nose_fall_dist = w / 12
        nose_thickness = max( w // 96, 1 )

        mouth_fall_dist = w // 32
        mouth_thickness = max( w // 64, 1 )

        eyes_mask = gdf('eyes',eyes_thickness)
        eyes_mask = 1-np.clip( eyes_mask/ eyes_fall_dist, 0, 1)

        brows_mask = gdf('brows', brows_thickness)
        brows_mask = 1-np.clip( brows_mask / brows_fall_dist, 0, 1)

        mouth_mask = gdf('mouth', mouth_thickness)
        mouth_mask = 1-np.clip( mouth_mask / mouth_fall_dist, 0, 1)

        def blend(a,b,k):
            x = np.clip ( 0.5+0.5*(b-a)/k, 0.0, 1.0 )
            return (a-b)*x+b - k*x*(1.0-x)

        nose_mask = blend ( gdf('up_nose', nose_thickness), gdf('down_nose', nose_thickness), nose_thickness*3 )
        nose_mask = 1-np.clip( nose_mask / nose_fall_dist, 0, 1)

        up_nose_mask = gdf('up_nose', nose_thickness)
        up_nose_mask = 1-np.clip( up_nose_mask / nose_fall_dist, 0, 1)

        down_nose_mask = gdf('down_nose', nose_thickness)
        down_nose_mask = 1-np.clip( down_nose_mask / nose_fall_dist, 0, 1)

        eyes_mask = eyes_mask * (1-mouth_mask)
        nose_mask = nose_mask * (1-eyes_mask)

        hull_mask = hull[...,0].copy()
        hull_mask = hull_mask * (1-eyes_mask) * (1-brows_mask) * (1-nose_mask) * (1-mouth_mask)


        mouth_mask= mouth_mask * (1-nose_mask)

        brows_mask = brows_mask * (1-nose_mask)* (1-eyes_mask )

        mask = mouth_mask + nose_mask + brows_mask  + eyes_mask

        rand_max_rotation_value = np.random.uniform(1, 5)
        rand_scaling = np.random.uniform(0.01, 0.05)
        rand_translation = np.random.uniform(0.5, 2.5)
        rand_args = [rand_max_rotation_value, rand_scaling, rand_translation]

        img_blended_eyes, _ = self.dynamic_blend(img.copy(), img, np.expand_dims(eyes_mask, axis=-1), landmark, rand_args)
        img_blended_nose, _ = self.dynamic_blend(img.copy(), img, np.expand_dims(nose_mask, axis=-1), landmark, rand_args)
        img_blended_mouth, _ = self.dynamic_blend(img.copy(), img, np.expand_dims(mouth_mask, axis=-1), landmark, rand_args)
        img_blended_eyebrows, _ = self.dynamic_blend(img.copy(), img, np.expand_dims(brows_mask, axis=-1), landmark, rand_args)  # Blend the eyebrows

        # Combine the blended images
        img_blended = cv2.addWeighted(img_blended_eyes, 0.25, img_blended_nose, 0.25, 0)
        img_blended = cv2.addWeighted(img_blended, 1, img_blended_mouth, 0.25, 0)
        img_blended = cv2.addWeighted(img_blended, 1, img_blended_eyebrows, 0.25, 0)  # Add the eyebrows to the blended image

        img_blended = img_blended.astype(np.uint8)
        img = img.astype(np.uint8)

        return img, img_blended, mask

    def dynamic_blend(self, source, target, mask, landmark, rand_args):
        rand_max_rotation_value, rand_scaling, rand_translation = rand_args
        
        # select three non-collinear points from the landmarks
        src_points = np.float32(
            self.perturbation_landmark(
                landmark, 
                rand_max_rotation_value, 
                rand_scaling, 
                rand_translation
            )[[30, 36, 45], :]
        )
        dst_points = np.float32(landmark[[30, 36, 45], :])

        # perform warp affine transformation
        warp_matrix = cv2.getAffineTransform(src_points, dst_points)
        warp_image = cv2.warpAffine(source, warp_matrix, (target.shape[1], target.shape[0]))

        # blend the warp image with the original image
        img_blended = (mask * warp_image + (1 - mask) * target)
        return img_blended, mask
    

    def perturbation_landmark(self, landmarks, max_rotation=5, max_scaling=0.05, max_translation=2):
        # Separate jaw landmarks and other landmarks
        jaw_landmarks = landmarks[:17]
        other_landmarks = landmarks[17:]

        # Compute the center of gravity of the other landmarks (excluding jaw)
        center = np.mean(other_landmarks, axis=0)

        # Compute the angle, scale and translation parameters
        angle = np.random.uniform(-max_rotation, max_rotation)
        scale = np.random.uniform(1 - max_scaling, 1 + max_scaling)
        translation = np.random.uniform(-max_translation, max_translation, size=2)

        # Create the transformation matrix
        transform = cv2.getRotationMatrix2D(tuple(center), angle, scale)
        transform[:, 2] += translation

        # Add a third row to the transformation matrix
        transform = np.vstack([transform, [0, 0, 1]])

        # Apply the transformation to the other landmarks
        perturbed_other_landmarks = np.dot(transform, np.column_stack([other_landmarks, np.ones(len(other_landmarks))]).T).T[:, :2]

        # Combine the jaw landmarks and the perturbed other landmarks
        perturbed_landmarks = np.vstack([jaw_landmarks, perturbed_other_landmarks])

        return perturbed_landmarks


def get_blend_mask(mask):
    H, W = mask.shape
    size_h = np.random.randint(192, 257)
    size_w = np.random.randint(192, 257)
    mask = cv2.resize(mask, (size_w, size_h))
    kernel_1 = random.randrange(5, 26, 2)
    kernel_1 = (kernel_1, kernel_1)
    kernel_2 = random.randrange(5, 26, 2)
    kernel_2 = (kernel_2, kernel_2)

    mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured[mask_blured < 1] = 0

    mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5, 46))
    mask_blured = mask_blured / (mask_blured.max())
    mask_blured = cv2.resize(mask_blured, (W, H))
    return mask_blured.reshape((mask_blured.shape + (1,)))
    
#     def self_blending(self, img, landmark):
#         H, W = len(img), len(img[0])
#         mask = np.zeros_like(img[:, :, 0])

#         # Create masks for individual facial features
#         eye_mask = np.zeros_like(mask)
#         nose_mask = np.zeros_like(mask)
#         mouth_mask = np.zeros_like(mask)
#         eyebrows_mask = np.zeros_like(mask)  # Added mask for eyebrows

#         # Fill the masks using the corresponding landmarks
#         cv2.fillConvexPoly(eye_mask, cv2.convexHull(landmark[36:48].astype(np.int32)), 1.)
#         cv2.fillConvexPoly(nose_mask, cv2.convexHull(landmark[27:36].astype(np.int32)), 1.)
#         cv2.fillConvexPoly(mouth_mask, cv2.convexHull(landmark[48:68].astype(np.int32)), 1.)
#         cv2.fillConvexPoly(eyebrows_mask, cv2.convexHull(np.concatenate((landmark[17:22], landmark[22:27])).astype(np.int32)), 1.)  # Fill the eyebrows mask

#         source = img.copy()

#         img_blended_eyes, _ = self.dynamic_blend(source, img, eye_mask, landmark)
#         img_blended_nose, _ = self.dynamic_blend(source, img, nose_mask, landmark)
#         img_blended_mouth, _ = self.dynamic_blend(source, img, mouth_mask, landmark)
#         img_blended_eyebrows, _ = self.dynamic_blend(source, img, eyebrows_mask, landmark)  # Blend the eyebrows

#         # Combine the blended images
#         img_blended = cv2.addWeighted(img_blended_eyes, 0.25, img_blended_nose, 0.25, 0)
#         img_blended = cv2.addWeighted(img_blended, 1, img_blended_mouth, 0.25, 0)
#         img_blended = cv2.addWeighted(img_blended, 1, img_blended_eyebrows, 0.25, 0)  # Add the eyebrows to the blended image

#         img_blended = img_blended.astype(np.uint8)
#         img = img.astype(np.uint8)

#         return img, img_blended, mask

#     def dynamic_blend(self, source, target, mask, landmark):
#         # select three non-collinear points from the landmarks
#         src_points = np.float32(self.perturbation_landmark(landmark)[[30, 36, 45], :])
#         dst_points = np.float32(landmark[[30, 36, 45], :])

#         # perform warp affine transformation
#         warp_matrix = cv2.getAffineTransform(src_points, dst_points)
#         warp_image = cv2.warpAffine(source, warp_matrix, (target.shape[1], target.shape[0]))

#         # blend the warp image with the original image
#         mask_blured = get_blend_mask(mask)
#         blend_list = [1, 1, 1]
#         blend_ratio = blend_list[np.random.randint(len(blend_list))]
#         mask_blured *= blend_ratio
#         img_blended = (mask_blured * warp_image + (1 - mask_blured) * target)
#         return img_blended, mask_blured
    

#     def perturbation_landmark(self, landmarks, max_rotation=5, max_scaling=0.05, max_translation=2):
#         # Separate jaw landmarks and other landmarks
#         jaw_landmarks = landmarks[:17]
#         other_landmarks = landmarks[17:]

#         # Compute the center of gravity of the other landmarks (excluding jaw)
#         center = np.mean(other_landmarks, axis=0)

#         # Compute the angle, scale and translation parameters
#         angle = np.random.uniform(-max_rotation, max_rotation)
#         scale = np.random.uniform(1 - max_scaling, 1 + max_scaling)
#         translation = np.random.uniform(-max_translation, max_translation, size=2)

#         # Create the transformation matrix
#         transform = cv2.getRotationMatrix2D(tuple(center), angle, scale)
#         transform[:, 2] += translation

#         # Add a third row to the transformation matrix
#         transform = np.vstack([transform, [0, 0, 1]])

#         # Apply the transformation to the other landmarks
#         perturbed_other_landmarks = np.dot(transform, np.column_stack([other_landmarks, np.ones(len(other_landmarks))]).T).T[:, :2]

#         # Combine the jaw landmarks and the perturbed other landmarks
#         perturbed_landmarks = np.vstack([jaw_landmarks, perturbed_other_landmarks])

#         return perturbed_landmarks


# def get_blend_mask(mask):
#     H, W = mask.shape
#     size_h = np.random.randint(192, 257)
#     size_w = np.random.randint(192, 257)
#     mask = cv2.resize(mask, (size_w, size_h))
#     kernel_1 = random.randrange(5, 26, 2)
#     kernel_1 = (kernel_1, kernel_1)
#     kernel_2 = random.randrange(5, 26, 2)
#     kernel_2 = (kernel_2, kernel_2)

#     mask_blured = cv2.GaussianBlur(mask, kernel_1, 0)
#     mask_blured = mask_blured / (mask_blured.max())
#     mask_blured[mask_blured < 1] = 0

#     mask_blured = cv2.GaussianBlur(mask_blured, kernel_2, np.random.randint(5, 46))
#     mask_blured = mask_blured / (mask_blured.max())
#     mask_blured = cv2.resize(mask_blured, (W, H))
#     return mask_blured.reshape((mask_blured.shape + (1,)))
        
    




class VBDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        assert mode=='train', 'vb only supports train mode'
        self.real_imglist = [img for img, label in zip(self.image_list, self.label_list) if label == 0]

        self.vb_api = VideoBlendAPI(phase='train', image_size=config['resolution'])
        self.clip_size = config['clip_size']
        self.transform = A.Compose([
            # A.RandomResizedCrop(config['resolution'], config['resolution'], scale=(0.8, 1.2), p=0.5),
            # A.Resize(config['resolution'], config['resolution']),
            # A.HorizontalFlip(),
            # A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            # A.HueSaturationValue(p=0.3),
            # A.ImageCompression(quality_lower=60, quality_upper=100, p=0.1),
            # A.GaussNoise(p=0.1),
            # A.MotionBlur(p=0.1),
            # A.CLAHE(p=0.1),
            # A.ChannelShuffle(p=0.1),
            # A.Cutout(p=0.1),
            # A.RandomGamma(p=0.3),
            # A.GlassBlur(p=0.3),
            A.Normalize(config['mean'], config['std']),
            ToTensorV2(),
        ])
    
    
    def __getitem__(self, index):
        image_paths = self.real_imglist[index]
        unique_video_name = image_paths[0].split('/')[-2]

        image_tensors = []
        vb_image_tensors = []
        landmark_tensors = []
        mask_tensors = []
        name_list = []
        fake_name_list = []

        for image_path in image_paths:
            if not os.path.exists(image_path):
                image_path = image_path.replace('/Youtu_Pangu_Security/public/', '/Youtu_Pangu_Security_Public/')


            # Get the mask and landmark paths
            landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')  # Use .npy for landmark

            # Load the image
            image = self.load_rgb(image_path)
            image = np.array(image)  # Convert to numpy array for data augmentation

            # Load mask and landmark
            landmarks = self.load_landmark(landmark_path)[:68]



            # ===== video-level blending ===== #
            # Apply the blend operation if the current frame is in the blend_frames set
            try:
                vb_image, _, blend_mask = self.vb_api(image, landmarks)
            except Exception as e:
                print(f"Error blending images at index {index}: {e}")
                return self.__getitem__(0)


            image_trans = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            vb_image_trans = cv2.cvtColor(vb_image, cv2.COLOR_BGR2RGB)


            image_tensors.append(image_trans)
            vb_image_tensors.append(vb_image_trans)
            landmark_tensors.append(None)
            mask_tensors.append(None)

        
        # os.makedirs('./vb_tmp/ori', exist_ok=True)
        # os.makedirs('./vb_tmp/vb', exist_ok=True)
        # for i in range(len(image_tensors)):
        #     cv2.imwrite('./vb_tmp/ori/image_{}.png'.format(i), image_tensors[i])
        #     cv2.imwrite('./vb_tmp/vb/vb_image_{}.png'.format(i), vb_image_tensors[i])

        # fuck

        # do video-level augmentation
        ###
        frames_tobe_auged = []
        frames_tobe_auged.extend(image_tensors)
        frames_tobe_auged.extend(vb_image_tensors)

        additional_targets = {}
        tmp_imgs = {'image': frames_tobe_auged[0]}
        for i in range(1, len(frames_tobe_auged)):
            additional_targets[f'image{i}'] = 'image'
            tmp_imgs[f'image{i}'] = frames_tobe_auged[i]
        self.transform.add_targets(additional_targets)
        transformed = self.transform(**tmp_imgs)
        transformed = OrderedDict(sorted(transformed.items(), key=lambda x: x[0]))
        transformed = list(transformed.values())
        
        real_tensors = torch.stack(transformed[:self.clip_size]).unsqueeze(0)
        fake_tensors = torch.stack(transformed[self.clip_size:]).unsqueeze(0)
        name_list.append(unique_video_name)
        fake_name_list.append(unique_video_name+'_'+'fake')
        
        return real_tensors, fake_tensors, name_list, fake_name_list

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Args:
            batch (list): A list of tuples containing the image tensor, the label tensor,
                          the landmark tensor, and the mask tensor.

        Returns:
            A tuple containing the image tensor, the label tensor, the landmark tensor,
            and the mask tensor.
        """
        # Separate the image, label, landmark, and mask tensors
        real_tensors, fake_tensors, name_list, fake_name_list = zip(*batch)
        real_tensors = torch.cat(real_tensors, dim=0)
        fake_tensors = torch.cat(fake_tensors, dim=0)
        images = torch.cat((real_tensors, fake_tensors), dim=0)
        real_labels = torch.zeros(len(name_list))
        fake_labels = torch.ones(len(fake_name_list))
        labels = torch.cat((real_labels, fake_labels), dim=0)
        names = name_list + fake_name_list

        # Create a dictionary of the tensors
        data_dict = {}
        data_dict['image'] = images
        data_dict['label'] = labels
        data_dict['landmark'] = None
        data_dict['mask'] = None
        data_dict['name'] = names
        return data_dict


    def __len__(self):
        return len(self.real_imglist)
    

if __name__ == '__main__':
    with open('/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/DeepfakeBenchv2/training/config/detector/i3d.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('./config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    config2['data_manner'] = 'lmdb'
    config['dataset_json_folder'] = '/Youtu_Pangu_Security/public/youtu-pangu-public/zhiyuanyan/DeepfakeBenchv2/preprocessing/dataset_json'
    config.update(config2)
    train_set = VBDataset(config=config, mode='train')
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