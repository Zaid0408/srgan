# Fix for the crop class in dataset.py
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import random

class mydata(Dataset):
    def __init__(self, LR_path, GT_path, in_memory=True, transform=None):
        self.LR_path = LR_path
        self.GT_path = GT_path
        self.in_memory = in_memory
        self.transform = transform
        
        # Sort and validate image pairs
        self.LR_img = sorted(os.listdir(LR_path))
        self.GT_img = sorted(os.listdir(GT_path))
        assert len(self.LR_img) == len(self.GT_img), "Mismatched LR/GT image counts"
        
        if in_memory:
            self.LR_img = [self._load_image(os.path.join(self.LR_path, lr)) for lr in self.LR_img]
            self.GT_img = [self._load_image(os.path.join(self.GT_path, gt)) for gt in self.GT_img]
    
    def _load_image(self, path):
        """Safe image loading with validation"""
        img = Image.open(path).convert('RGB')
        img = np.array(img).astype(np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Invalid image dimensions at {path}: {img.shape}")
        return img
        
    def __len__(self):
        return len(self.LR_img)
        
    def __getitem__(self, idx):
        try:
            if self.in_memory:
                GT = self.GT_img[idx].astype(np.float32)
                LR = self.LR_img[idx].astype(np.float32)
            else:
                GT = self._load_image(os.path.join(self.GT_path, self.GT_img[idx]))
                LR = self._load_image(os.path.join(self.LR_path, self.LR_img[idx]))

            # Normalize and create sample
            sample = {
                'GT': (GT / 127.5) - 1.0,
                'LR': (LR / 127.5) - 1.0
            }

            if self.transform:
                sample = self.transform(sample)
                
            # Ensure correct tensor shape (C, H, W)
            sample['GT'] = np.ascontiguousarray(sample['GT'].transpose(2, 0, 1))
            sample['LR'] = np.ascontiguousarray(sample['LR'].transpose(2, 0, 1))
            
            return sample
            
        except Exception as e:
            print(f"Error processing image {idx}: {e}")
            # Return next valid image
            return self.__getitem__((idx + 1) % len(self))
# class mydata(Dataset):
#     def __init__(self, LR_path, GT_path, in_memory = True, transform = None):
        
#         self.LR_path = LR_path
#         self.GT_path = GT_path
#         self.in_memory = in_memory
#         self.transform = transform
        
#         self.LR_img = sorted(os.listdir(LR_path))
#         self.GT_img = sorted(os.listdir(GT_path))
        
#         if in_memory:
#             self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr)).convert("RGB")).astype(np.uint8) for lr in self.LR_img]
#             self.GT_img = [np.array(Image.open(os.path.join(self.GT_path, gt)).convert("RGB")).astype(np.uint8) for gt in self.GT_img]
        
#     def __len__(self):
#         return len(self.LR_img)
        
#     def __getitem__(self, i):
#         try:
#             img_item = {}
            
#             if self.in_memory:
#                 GT = self.GT_img[i].astype(np.float32)
#                 LR = self.LR_img[i].astype(np.float32)
                
#             else:
#                 GT = np.array(Image.open(os.path.join(self.GT_path, self.GT_img[i])).convert("RGB"))
#                 LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])).convert("RGB"))

#             # Validate dimensions
#             if LR.shape[0] == 0 or LR.shape[1] == 0 or GT.shape[0] == 0 or GT.shape[1] == 0:
#                 print(f"Warning: Zero dimension image at index {i}")
#                 # Try next image if this one has issues
#                 return self.__getitem__((i + 1) % len(self))

#             img_item['GT'] = (GT / 127.5) - 1.0
#             img_item['LR'] = (LR / 127.5) - 1.0
                    
#             if self.transform is not None:
#                 try:
#                     img_item = self.transform(img_item)
#                 except Exception as e:
#                     print(f"Transform error at index {i}: {e}")
#                     # Try next image if transform fails
#                     return self.__getitem__((i + 1) % len(self))
                
#             img_item['GT'] = img_item['GT'].transpose(2, 0, 1).astype(np.float32)
#             img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
            
#             # Final check for zero dimensions after transforms
#             if 0 in img_item['GT'].shape or 0 in img_item['LR'].shape:
#                 print(f"Warning: Zero dimension in tensor at index {i} after transforms")
#                 return self.__getitem__((i + 1) % len(self))
                
#             return img_item
            
#         except Exception as e:
#             print(f"Error processing image at index {i}: {e}")
#             # Try next image
#             return self.__getitem__((i + 1) % len(self))


class testOnly_data(Dataset):
    # Test data class remains unchanged
    def __init__(self, LR_path, in_memory = True, transform = None):
        
        self.LR_path = LR_path
        self.LR_img = sorted(os.listdir(LR_path))
        self.in_memory = in_memory
        if in_memory:
            self.LR_img = [np.array(Image.open(os.path.join(self.LR_path, lr))) for lr in self.LR_img]
        
    def __len__(self):
        return len(self.LR_img)
        
    def __getitem__(self, i):
        img_item = {}
        
        if self.in_memory:
            LR = self.LR_img[i]
            
        else:
            LR = np.array(Image.open(os.path.join(self.LR_path, self.LR_img[i])))

        img_item['LR'] = (LR / 127.5) - 1.0                
        img_item['LR'] = img_item['LR'].transpose(2, 0, 1).astype(np.float32)
        
        return img_item

class crop(object):
    def __init__(self, scale, patch_size):
        self.scale = scale
        self.patch_size = patch_size
        
    def __call__(self, sample):
        LR, GT = sample['LR'], sample['GT']
        
        # Validate input shapes
        h, w = LR.shape[:2]
        if h < self.patch_size or w < self.patch_size:
            # Pad if too small
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            LR = np.pad(LR, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            GT = np.pad(GT, ((0, pad_h*self.scale), (0, pad_w*self.scale), (0, 0)), mode='reflect')
            h, w = LR.shape[:2]

        # Random crop coordinates
        ix = random.randint(0, w - self.patch_size)
        iy = random.randint(0, h - self.patch_size)
        
        # Calculate HR crop coordinates
        tx = ix * self.scale
        ty = iy * self.scale
        
        # Extract patches
        LR_patch = LR[iy:iy+self.patch_size, ix:ix+self.patch_size]
        GT_patch = GT[ty:ty+self.scale*self.patch_size, tx:tx+self.scale*self.patch_size]
        
        # Final validation
        assert LR_patch.shape[:2] == (self.patch_size, self.patch_size), \
            f"Invalid LR patch shape: {LR_patch.shape}"
        assert GT_patch.shape[:2] == (self.scale*self.patch_size, self.scale*self.patch_size), \
            f"Invalid GT patch shape: {GT_patch.shape}"
            
        return {'LR': LR_patch, 'GT': GT_patch}

# class crop(object):
#     def __init__(self, scale, patch_size):
#         self.scale = scale
#         self.patch_size = patch_size
        
#     def __call__(self, sample):
#         LR_img, GT_img = sample['LR'], sample['GT']
#         ih, iw = LR_img.shape[:2]
        
#         # FIXED: Make sure the image is large enough for the patch
#         if ih <= self.patch_size or iw <= self.patch_size:
#             # If image is too small, return the whole image (no crop)
#             # and pad if necessary to reach patch_size
#             if ih < self.patch_size:
#                 pad_h = self.patch_size - ih
#                 LR_img = np.pad(LR_img, ((0, pad_h), (0, 0), (0, 0)), mode='reflect')
#                 GT_img = np.pad(GT_img, ((0, pad_h*self.scale), (0, 0), (0, 0)), mode='reflect')
            
#             if iw < self.patch_size:
#                 pad_w = self.patch_size - iw
#                 LR_img = np.pad(LR_img, ((0, 0), (0, pad_w), (0, 0)), mode='reflect')
#                 GT_img = np.pad(GT_img, ((0, 0), (0, pad_w*self.scale), (0, 0)), mode='reflect')
            
#             # Use the whole padded image
#             LR_patch = LR_img[:self.patch_size, :self.patch_size]
#             GT_patch = GT_img[:self.scale*self.patch_size, :self.scale*self.patch_size]
#         else:
#             # Normal cropping for images larger than patch_size
#             max_x = max(0, iw - self.patch_size)
#             max_y = max(0, ih - self.patch_size)
            
#             ix = random.randrange(0, max_x + 1)
#             iy = random.randrange(0, max_y + 1)
            
#             tx = ix * self.scale
#             ty = iy * self.scale
            
#             LR_patch = LR_img[iy:iy + self.patch_size, ix:ix + self.patch_size]
#             GT_patch = GT_img[ty:ty + (self.scale * self.patch_size), tx:tx + (self.scale * self.patch_size)]
        
#         # Final validation - ensure crops have the right shape
#         if LR_patch.shape[0] != self.patch_size or LR_patch.shape[1] != self.patch_size:
#             raise ValueError(f"LR patch has incorrect size: {LR_patch.shape}")
            
#         if GT_patch.shape[0] != self.scale*self.patch_size or GT_patch.shape[1] != self.scale*self.patch_size:
#             raise ValueError(f"GT patch has incorrect size: {GT_patch.shape}")
            
#         return {'LR': LR_patch, 'GT': GT_patch}


# class augmentation(object):
#     # Augmentation class remains unchanged
#     def __call__(self, sample):
#         LR_img, GT_img = sample['LR'], sample['GT']
        
#         hor_flip = random.randrange(0,2)
#         ver_flip = random.randrange(0,2)
#         rot = random.randrange(0,2)
    
#         if hor_flip:
#             temp_LR = np.fliplr(LR_img)
#             LR_img = temp_LR.copy()
#             temp_GT = np.fliplr(GT_img)
#             GT_img = temp_GT.copy()
            
#             del temp_LR, temp_GT
        
#         if ver_flip:
#             temp_LR = np.flipud(LR_img)
#             LR_img = temp_LR.copy()
#             temp_GT = np.flipud(GT_img)
#             GT_img = temp_GT.copy()
            
#             del temp_LR, temp_GT
            
#         if rot:
#             LR_img = LR_img.transpose(1, 0, 2)
#             GT_img = GT_img.transpose(1, 0, 2)
        
#         return {'LR': LR_img, 'GT': GT_img}
class augmentation(object):
    def __call__(self, sample):
        LR, GT = sample['LR'], sample['GT']
        
        # Horizontal flip
        if random.random() > 0.5:
            LR = np.fliplr(LR).copy()
            GT = np.fliplr(GT).copy()
            
        # Vertical flip
        if random.random() > 0.5:
            LR = np.flipud(LR).copy()
            GT = np.flipud(GT).copy()
            
        # Rotation (90° only to maintain shape)
        if random.random() > 0.5:
            LR = np.rot90(LR, 1, (0, 1)).copy()
            GT = np.rot90(GT, 1, (0, 1)).copy()
            
        return {'LR': LR, 'GT': GT}
