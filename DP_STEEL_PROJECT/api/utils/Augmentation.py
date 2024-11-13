
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.preprocessing.image import load_img, save_img
import numpy as np
import argparse
import albumentations as A
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import PIL
import shutil

import helpers


class Augmentation:
    '''
    Class representing image augmentation.
    
    This class provides a basic structure for augmenting images.
    
    Attributes:
        input_path (str): Path to the folder containing images for augmentation.
        transforms (list): List of transformations.
        output (str): Path to the folder where augmented images will be saved.
        images (list): List of images paths in the input folder.
        files (list): List of full paths to images in the input folder.
    '''
    
    def __init__(self, args: argparse.Namespace = None, preprocessor=None) -> None:
        '''Initializes the Augmentation class with the input path and optional preprocessor.'''
        if args is not None:
            self.input_path = args.input
        elif preprocessor is not None:
            self.input_path = preprocessor.output
        else:
            raise ValueError("Please provide input path or preprocessor.")
        
        self.transforms = [
            #A.GaussianBlur(blur_limit=3, p=1),
            #A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            #A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
            #flip -90 90 180 degrees
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.Transpose(p=1),
        ]
        self.output = os.path.join(self.input_path, "augmented")
        helpers.create_if_not_exists(self.output, clear=True)
        self.images = helpers.list_all_images(self.input_path)
        #full path to images in input folder
        self.files = [os.path.join(self.input_path, x) for x in os.listdir(self.input_path)]
        #check wich images are not in images but exist in self.files
        #self.excluded = [img for img in self.files if img not in self.images]
        self.apply_transforms() 
        
    def orthogonal_rot(self, image: PIL.Image.Image) -> list[PIL.Image.Image]:
        '''Returns:   list[PIL.Image.Image]: List of rotations of the image by 90, 180, 270 degrees'''
        return  [np.rot90(image, i) for i in range(1,4)]
    
    def apply_rotations(self, img_path: str) -> None:
        '''Applies orthogonal rotations to the image in img_path.'''
        image = load_img(img_path)
        augmented = self.orthogonal_rot(image)
        filename, ext = os.path.splitext(img_path)
        filename = os.path.basename(filename)
        for i, img in enumerate(augmented):
            save_img(self.output+os.sep+f'{filename}_aug_Rot_{(i+1)*90}{ext}', img)
            
    def apply_transforms(self) -> None:
        '''Applies transformations to the images. Saves the transformed images to the output folder.'''
        print('Applying transforms...')
        for img_path in tqdm(self.images):
            shutil.move(img_path, img_path.replace("°", "_deg_"))
            img_path = img_path.replace("°", "_deg_")
            img = cv2.imread(img_path)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            grid = {}
            grid['original'] = {
                'image': img,
                'path': img_path
            }
            for transform in self.transforms:
                augmented_image = transform(image=img)['image']
                grid[transform.__class__.__name__] = {
                    'image': augmented_image,
                    'path': img_path
                }
            #print(image_filename)
            #visualize(grid)
            self.save_transforms(grid)
            #self.apply_rotations(img_path)
    
    def save_transforms(self, transforms_dict: dict) -> None:
        '''Saves the transformed images to the output folder.'''
        for transform_name, image in transforms_dict.items():
            file_name, file_extension = os.path.splitext(image['path'])
            file_name = os.path.basename(file_name)
            if transform_name == 'original':
                save_path = os.path.join(self.output, file_name + file_extension)
            else:
                save_path = os.path.join(self.output, file_name + '_aug_' + transform_name + file_extension)
            #print(save_path)
            cv2.imwrite(save_path, cv2.cvtColor(image['image'], cv2.COLOR_RGB2BGR))
    
    def visualize(self, transforms_dict: dict) -> None:
        '''Visualizes the original image and its transformations.'''
        original_img = transforms_dict['original']['image']
        fig = plt.figure(figsize=(20, 10))
        for i, (transform_name, image) in enumerate(transforms_dict.items()):
            ax = fig.add_subplot(3, 3, i + 1)
            ax.imshow(original_img - image['image'])
            ax.title.set_text(f'{transform_name}')
            ax.axis('off')
            plt.show()   


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-i",
        "--input",
        required=False,
        help="path to folder with images for augmentation",
    )
    args = ap.parse_args()
    aug = Augmentation(args)


if __name__ == "__main__":
    main()
