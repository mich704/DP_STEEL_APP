import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import argparse
import logging
logger = logging.getLogger('api.utils')

try:
    from helpers import create_if_not_exists, list_all_images
except:
    from .helpers import create_if_not_exists, list_all_images


from django.conf import settings as PROJECT_SETTINGS

from api.models import PreprocessedImage, Publication, ExtractedImage, ClassifiedImage, Image
from tqdm import tqdm


class SimplePreprocessor():
    '''
    Class representing preprocessing of images extracted from scientific publications.
    As an input it takes a path to the folder containing images that are NOT classfied yet.
    
    This class provides a basic structure for preprocessing images such as:
        - splitting single image object with many subimages to granular subimages.
    
    Attributes:
        input (str): Path to the folder containing images to split.
        output (str): Path to the folder where preprocessed images will be saved.
        sobel (bool): Use sobel filter for edge detection.
        split (bool): Split images into subimages.
        img_list (list): List of images in the input folder.
        show (bool): Show images in matplotlib.
        save (bool): Save images to output folder.
        db_manager (DatabaseManagerAlchemy): Database manager.
    '''
    
    def __init__(self, args: argparse.Namespace, publication: Publication = None) -> None:
        '''Initializes the Preprocessing class with the input arguments.'''
        if args.input is None:
            raise ValueError("Input path is required.")
        self.input = args.input
        self.output = (
            args.output
            if args.output is not None
            else os.path.join(self.input, "preprocessed")
        )
        self.publication = publication
        if self.publication is not None:
            self.output_folder = os.path.join(PROJECT_SETTINGS.IMAGES_DIR, 'preprocessed', os.path.splitext(self.publication.filename)[0])
        else:
            self.output_folder = os.path.join(PROJECT_SETTINGS.IMAGES_DIR, 'preprocessed')
        self.input_img_list = list_all_images(self.input)
        create_if_not_exists(self.output, clear=True)
        if len(self.input_img_list) > 0:
            logger.info("\nSplitting extracted images into subimages:")
            for i, img_path in tqdm(enumerate(self.input_img_list), total=len(self.input_img_list)):
                filename = os.path.basename(img_path)
                self.preprocess(img_path)
            #logger.debug(f'Splitting done. Saved into {self.output}')
        else:
            print("No images to split. Exiting...")
                
    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        '''
        Parses only known command line arguments.
        Returns:
            argparse.Namespace: Parsed arguments.
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input", required=False, help="path to folder with images to preprocess")
        parser.add_argument("-o", "--output", required=False, help="path to output folder")
        parser.add_argument("--sobel", required=False, help="find subimages contours using sobel", action="store_true")
        parser.add_argument("--split", required=False, action="store_true",
                             help="split microstructures sub-images by vertical and horizontal edges")
        parser.add_argument("--save", required=False, help="save sub-images to output folder", action="store_true")
        parser.add_argument("--show", required=False, help="show sub-images in matplotlib", action="store_true")
        args, unknown = parser.parse_known_args()
        return argparse.Namespace(**vars(args))
    
    def preprocess(self, img_path: str) -> None:
        self.split_to_subimages(img_path)
                
    def is_rectangle_contained(self, rectangle: tuple[int], rectangles: list[tuple]) -> bool:
        '''
        Check if rectangle is contained within any of rectangles +- 10 pixels
        Args:
            rectangle: tuple(x_start, y_start, x_end, y_end) - rectangle to check
            rectangles: list of tuples [(x_start, y_start, x_end, y_end), ...] - list of rectangles
        '''
        for r in rectangles:
            if rectangle[0] >= r[0] and rectangle[2] <= r[2]-10 and rectangle[1] >= r[1] and rectangle[3] <= r[3]-10:
                return True
        return False
    
    def threshold_image(self, original_img: np.ndarray, color_threshold: int = 30) -> np.ndarray:
        '''
        Thresholding grayscale image background to black
        
        Args:
            original_img: (np.ndarray) - original image
            color_threshold: (int) - threshold value for color
        Returns:
            np.ndarray - thresholded image
        '''
        tmp = original_img.copy()
        img_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        img_gray[img_gray < color_threshold] = 0
        img_gray[img_gray > 255 - color_threshold] = 0
        # Thresholding indices for white and black pixels
        w = img_gray == 255
        b = img_gray == 0
        tmp[w] = [0, 0, 0]
        tmp[b] = [0, 0, 0]
        return tmp
    
    def find_subimages_contours(self, original_img: np.ndarray, thresholded_img: np.ndarray) -> list[np.ndarray]:
        '''
        Find contours of subimages in thresholded image
        
        Args:
            original_img: (np.ndarray) - original image
            thresholded_img: (np.ndarray) - thresholded image
        Returns:
            list[np.ndarray] - list of subimages [np.ndarray, ...]
        '''
        # Pixels with intensity values less than 12 will be set to 0 (black), 
        # and pixels with intensity values greater than or equal to 12 will be set to 255 (white).
        img_copy = thresholded_img.copy()
        gray = cv2.cvtColor(thresholded_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
  
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
        subimages = []
        subimages_rectangles = [
                (x, y, x + w, y + h) for x, y, w, h 
                in [cv2.boundingRect(contour) for contour in contours]
                if w >= 50 and h >= 50]
        subimages_rectangles = [
                rectangle for rectangle 
                in subimages_rectangles
                if not self.is_rectangle_contained(rectangle, subimages_rectangles)]
        
        for xd, rectangle in enumerate(subimages_rectangles):
            x, y, w, h = rectangle
            w, h = w - x, h - y
           
            subimage = original_img[y : y + h, x : x + w]
            #cv2.rectangle(thresholded_img, (x, y), (x + w, y + h), (0,255,0), 5)
            #subimages.append(subimage)
            #     subimages.append(subimage)
            # #check if subimage is bigger than 95% of the original image
            #if subimage.shape[0] > 0.95 * original_img.shape[0] and subimage.shape[1] > 0.95 * original_img.shape[1]:
                #TODO fix setting subimages to [] and return, does only work on preprocessed images (2nd iteration)
                #crop 20px from the edges
                #subimage = subimage[20:-20, 20:-20]
                #thresh = self.threshold_image(subimage, 20)
                #print("Subimage is bigger than 95% of the original image. Splitting it into subimages.")
                #return self.find_subimages_contours(subimage, thresh)  
            #else:
            cv2.rectangle(thresholded_img, (x, y), (x + w, y + h), (0,255,0), 5)
            subimages.append(subimage)
    
        return subimages
        
    def split_to_subimages(self, img_path: str, color_threshold: int = 30) -> None:
        """
        Algorithm to split image containg subimages into signle subimage files.
        
        Args:
            img_path: (str) - path to the image
            color_threshold: (int) - threshold value for color
        """
        # Read the image
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        thresholded = self.threshold_image(img, color_threshold)
        subimages = self.find_subimages_contours(img, thresholded)
        #TODO redo this part, too much hardcoded values
        while len(subimages) >= 8 & color_threshold > 0:
            color_threshold -= 10
            thresholded = self.threshold_image(img, color_threshold)
            subimages = self.find_subimages_contours(img, thresholded)
       
        num_subplots = len(subimages) + 1
        if num_subplots > 2:
            for j, subimage in enumerate(subimages):
                base_name, extension = os.path.splitext(filename)
                preprocessed_img_data = {
                    'path': os.path.join(self.output,f"{base_name}_sub_{j}{extension}"),
                    'label': None,
                    'parent_image_path': img_path
                }
                self.save_to_database(f"{base_name}_sub_{j}{extension}", subimage, img_path)
          
        else:
            self.save_to_database(filename, img, img_path)
           
      

    def save_to_database(self, filename: str, img: np.ndarray, parent_img_path: str = None) -> None:
        '''Save given image with filename to self.output folder and into.'''
        path = os.path.join(self.output, filename)
        try:
            ex_img_parent_id = ExtractedImage.objects.get(path=parent_img_path).id
            if not PreprocessedImage.objects.filter(path=path,
                                            extracted_image_parent_id=ex_img_parent_id).exists():
                PreprocessedImage.objects.create(path=path, 
                                                 extracted_image_parent_id=ex_img_parent_id)
            cv2.imwrite(path, img)
         
       
        except ExtractedImage.DoesNotExist:
            raise ValueError("Parent image does not exist in the database.")
        
    def get_images_paths(self) -> list[str]:
        '''Return list of paths to preprocessed images.'''
        extracted_images = ExtractedImage.objects.filter(publication=self.publication)
        return [img.path for img in PreprocessedImage.objects.filter(extracted_image_parent__in=extracted_images)]
        
        
def main():
    #5817
    # C:\Users\Michal\Desktop\DP-STEEL-AI_v2\storage\images\extracted\s12613-023-2751-1\preprocessed\classification\image_classifier_CNN\rest\s12613-023-2751-1_page8_img130.jpeg
    # rest
    # 6604
    path = r'C:\Users\kraft\Desktop\DP_STEEL_AI\DP-STEEL-AI_v2\image.png'
    preprocessor = SimplePreprocessor()
    preprocessor.preprocess(path)
    pass
   
   
    
    

if __name__ == "__main__":
    main()

#plt.imshow(tmp)
# v_height = v_props['peak_heights'] #list of the heights of the peaks
# h_height = h_props['peak_heights'] #list of the heights of the peaks

# def align_axis_x(ax, ax_target):
#     """Make x-axis of `ax` aligned with `ax_target` in figure"""
#     posn_old, posn_target = ax.get_position(), ax_target.get_position()
#     ax.set_position([posn_target.x0, posn_old.y0, posn_target.width, posn_old.height])

# def align_axis_y(ax, ax_target):
#     """Make y-axis of `ax` aligned with `ax_target` in figure"""
#     posn_old, posn_target = ax.get_position(), ax_target.get_position()
#     ax.set_position([posn_old.x0, posn_target.y0, posn_old.width, posn_target.height])

# fig = plt.figure(constrained_layout=False, figsize=(24,16))
# spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 1])
# ax1 = fig.add_subplot(spec[0,0])

# ax2 = fig.add_subplot(spec[0, 1])
# ax2.imshow(v_morphed)
# ax3 = fig.add_subplot(spec[0, 2])
# ax3.imshow(h_morphed)
# ax4 = fig.add_subplot(spec[0, 3], sharey=ax3)
# ax4.plot(h_acc[:,0], np.arange(len(h_acc[:,0])), 'y', marker="o", ms=1, mfc="k", mec="k")
# ax4.plot(s_h_acc, np.arange(len(s_h_acc)), 'r', lw=1)
# ax4.plot(h_height, h_peaks, "x", lw="5")
# ax5 = fig.add_subplot(spec[1, 1], sharex=ax2)
# ax5.plot(np.arange(len(v_acc[0,:])), v_acc[0,:], 'y', marker="o", ms=1, mfc="k", mec="k")
# ax5.plot(np.arange(len(s_v_acc)), s_v_acc, 'r', lw=2)
# ax5.plot(v_peaks, v_height, "x", lw="5")
# plt.tight_layout()
# plt.show()
# align_axis_y(ax4,ax3)
# align_axis_x(ax5,ax2)
'''
##detect horizontal and vertical lines using cv2.HoughLinesP
edges = cv2.Canny(mask, 12, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
#filter out horizontal and vertical lines in oneliner
horizontal_lines = [line for line in lines if abs(line[0][1] - line[0][3]) < 10]
vertical_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 10]
#merge horizontal and vertical lines
lines = np.concatenate((horizontal_lines, vertical_lines))
#draw lines on the image
#if line is longer than 95% of the image width or height red color
for line in lines:
    x1, y1, x2, y2 = line[0]
    if abs(x1 - x2) > 0.95 * mask.shape[1] or abs(y1 - y2) > 0.95 * mask.shape[0]:
        #set this line to white on mask image
        cv2.line(mask, (x1, y1), (x2, y2), (255, 255, 255), 1)
        #if pixel below and above the line is black set it to black
        if mask[y1 + 2, x1] == 255 and mask[y1 - 2, x1] == 255:
            mask[y1, x1] = 255
        
        cv2.line(img_copy, (x1, y1), (x2, y2), (255, 255, 255), 2)
    else:
        cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
'''