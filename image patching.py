#!/usr/bin/env python
# coding: utf-8

# In[21]:


import os
import random


# In[22]:


import cv2


# In[23]:


pip install opencv-python


# In[7]:


import cv2


# In[24]:


img_ben=r'C:\Users\user\Desktop\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria\400X\benign'
img_mal=r'C:\Users\user\Desktop\BreakHis - Breast Cancer Histopathological Database\dataset_cancer_v1\dataset_cancer_v1\classificacao_binaria\400X\malignant'
image_ben=[file for file in os.listdir(img_ben)]
image_mal=[file for file in os.listdir(img_mal)]


# In[25]:


print(len(image_ben))
print(len(image_mal))


# In[26]:


def find_interest_points(image):
    
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints


# In[11]:


def find_interest_points_op(image):
    # Create an ORB object
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    # Draw keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Annotate keypoints with their coordinates
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(image_with_keypoints, (int(x), int(y)), 1, (0, 255, 0), -1)  # Draw a point
    
    return image_with_keypoints


# In[13]:


image = cv2.imread(r"C:\Users\user\Desktop\New folder\malignant\SOB_M_DC-14-2523-400-004.png")

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find interest points and display them on the original image
image_with_keypoints = find_interest_points(gray_image)

# Display the image with keypoints
cv2.imshow("Image with Keypoints", image_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[27]:


patch_size = 128
fin_patches_ben=[]
fin_patches_mal=[]
num_patches_ben=[]
num_patches_mal=[]


# In[28]:


def get_patches(patch_size,keypoints, image):#will return list of patches and number of patches for an image
    patches = []
    
    
    for keypoint in keypoints:
        x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
        if x-patch_size//2<0 or x+patch_size//2>image.shape[1] or y-patch_size//2<0 or y+patch_size>image.shape[0]:
            pass
        else:
            
            patch = image[y - patch_size // 2:y + patch_size // 2, x - patch_size // 2:x + patch_size // 2]
            patches.append(patch)
    
    return patches,len(patches)
    
    
    





# In[29]:


#if i have a list of patches, how to make them all equal?
#i can try data augmentation, oversampling or try generating  a loss function that gives more weight to clas with less number of patches


# In[30]:


for file in image_ben:
    img_path = os.path.join(img_ben, file)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    keypoints=find_interest_points(image)
    patches,p=get_patches(patch_size,keypoints, image)
    fin_patches_ben.extend(patches)
    num_patches_ben.append(p)


# In[31]:


print(len(fin_patches_ben))
print(len(num_patches_ben))


# In[32]:


for file in image_mal:
    img_path = os.path.join(img_mal, file)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    keypoints=find_interest_points(image)
    patches,p=get_patches(patch_size,keypoints, image)
    fin_patches_mal.extend(patches)
    num_patches_mal.append(p)


# In[33]:


print(len(fin_patches_mal))
print(len(num_patches_mal))


# In[27]:


pip install imbalanced-learn


# In[30]:


get_ipython().system('pip install --upgrade scikit-learn')
get_ipython().system('pip install --upgrade imbalanced-learn')


# In[24]:


#it is almost 2:1
#trying smote

from imblearn.over_sampling import SMOTE
max_patches = max(len(fin_patches_mal), len(num_patches_mal))
oversampled_patches, _ = smote.fit_resample(np.array(fin_patches_ben), np.zeros(len(fin_patches_ben)))
fin_patches_ben.extend(oversampled_patches)


# In[34]:


#random oversampling
import os
import cv2
import numpy as np
from sklearn.utils import resample



min_class_patches = min(len(fin_patches_ben), len(fin_patches_mal))

# Perform random oversampling on the minority class (benign)
fin_patches_ben_oversampled = resample(fin_patches_ben, replace=True, n_samples=len(fin_patches_mal)-len(fin_patches_ben))

# Concatenate the oversampled data with the original data
fin_patches_ben_balanced = np.concatenate([fin_patches_ben, fin_patches_ben_oversampled])


# In[41]:


#len(fin_patches_ben_balanced)
fin_patches_mal_np=np.array(fin_patches_mal)
print(fin_patches_mal_np.shape)
print(fin_patches_ben_balanced.shape)


# In[36]:


import matplotlib.pyplot as plt
import numpy as np


# In[42]:


plt.imshow(fin_patches_mal_np[20000])
plt.axis('off')  # Turn off axis
plt.show()


# In[43]:


import cv2
import os

def save_image_patches(image_patches_list, output_folder):
    
    # Iterate through the image patches and save them to the output folder
    for i, image_patch in enumerate(image_patches_list):
        # Define the file path for saving
        file_path = os.path.join(output_folder, f"image_patch_{i}.png")
        
        # Save the image patch
        cv2.imwrite(file_path, image_patch)


# Specify the output folder
output_folder1 = r"C:\Users\user\Desktop\image_patches_benign"

# Save image patches from list1
save_image_patches(fin_patches_ben_balanced, output_folder1)

# Save image patches from list2



# In[45]:


output_folder2=r"C:\Users\user\Desktop\image_patches_mal"
save_image_patches(fin_patches_mal_np, output_folder2)


# In[ ]:




