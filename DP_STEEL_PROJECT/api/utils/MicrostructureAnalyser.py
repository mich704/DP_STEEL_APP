import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import os
import argparse
import re
#import easyocr
import math
import pandas as pd

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

from ..storage import MediaStorage

from .helpers import create_if_not_exists, list_all_images, RGB_COLORS
import logging
logger = logging.getLogger('api.utils')
from tqdm import tqdm



MICROSTRUCTURE_ANALYSIS_STORAGE = MediaStorage(subdir='microstructure_analysis')

MARTENSITE_PROPERITES = {
    'young_modulus': [207, 'GPa'],
    'yield_stress': [800, 'MPa'],
    'ultimate_tensile_strength': [1.2, 'GPa'],
}

FERRITE_PROPERITES = {
    'young_modulus': [200, 'GPa'],
    'yield_stress': [300, 'MPa'],
    'ultimate_tensile_strength': [0.5, 'GPa'],
}

class MicrostructureAnalyser():
    '''
    Class representing preprocessing and analysis of microstructure images extracted from scientific publications.
    As an input it takes a path to the folder containing images that are classfied as microstructures.
    #TODO
    1. Improve thresholding approach
    2. Implement some unsupervised learning approach to detect grain boundaries 
        (kmeans, DBSCAN , AgglomerativeClustering, etc.)
        
    Attrs:
        images_list: (list) - list of images to segment
        session_id: (str) - unique session id for storing the results
    '''
    def __init__(self, images_list, session_id) -> None:
        args = self.parse_arguments()
        self.images_list = images_list
        self.session_id = session_id
        self.output_files = []
        self.martensite_threshold = 0.61
        self.cmap = plt.get_cmap('RdGy')
        self.output = os.path.join(MICROSTRUCTURE_ANALYSIS_STORAGE.base_location, f'{self.session_id}')
        create_if_not_exists(self.output)
        self.csv_raport = os.path.join(self.output, 'microstructure_analysis.csv')
        self.create_csv_template()
        
    def run(self) -> None:
        try:
            if len(self.images_list) > 0:
                channel_layer = get_channel_layer()
                logger.info("\nSegmentation of microstructure images:")
                for i, img_file in tqdm(enumerate(self.images_list), total=len(self.images_list)):
                    self.preprocess(img_file.name)  
                    async_to_sync(channel_layer.group_send)(
                        self.session_id,
                        {
                            'type': 'progress_update',
                            'message': f'Processing file {i+1}/{len(self.images_list)}'
                        }
                    )
                logger.debug(f'\nSegmentation done. Saved into {self.output}')
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.save_to_xlsx()
        except Exception as e:
            raise e
        finally:
            self.save_to_xlsx()    
        
    # def get_img_micrometer_value(self, img_path: str) -> str:
    #     img = cv2.imread(img_path)  
    #     ocr_result = self.ocr_with_easyocr(img)
    #     micrometer_value = self.extract_micrometer_value(ocr_result)
    #     # if micrometer_value is not None:
        #     #add mictometer value to the image
        #     cv2.putText(img, micrometer_value, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #     img_name, extension = os.path.splitext(os.path.basename(img_path))
        #     output_path = os.path.join(self.output,  img_name + '_micrometer' + extension)
        #     self.save_to_output(output_path, img)
    
    def create_csv_template(self) -> None:
        '''Create csv template for mechanical properties.'''
        columns = ['Image', 'Martensite (%)', 'Ferrite (%)',
                   'Boundaries (%)', 'Mean E (GPa)',
                   'Mean σY (MPa)', 'Mean UTS (GPa)']
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.csv_raport, index=False)
        self.output_files.append(self.csv_raport)
        
    def save_to_xlsx(self) -> None:
        '''Save mechanical properties to xlsx file.'''
        if self.csv_raport is not None:
            import xlsxwriter
            df = pd.read_csv(self.csv_raport)
            # Convert the CSV file to an Excel file
            df.to_excel(os.path.join(self.output, 'microstructure_analysis.xlsx'), index=False)
            writer = pd.ExcelWriter(os.path.join(self.output, 'microstructure_analysis.xlsx'), engine='xlsxwriter')

            # Write the DataFrame to the Excel file
            df.to_excel(writer, index=False, sheet_name='raport')
            # Access the workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets['raport']
            # Format the "Image" column as hyperlinks
            for idx, col in enumerate(df.columns):
                # Calculate the maximum length in the column and set the width
                column_len = df[col].astype(str).map(len).max()
                column_len = max(column_len, len(col)) + 0.5  # Add some extra space
                worksheet.set_column(idx, idx, column_len)

                # Add hyperlinks for the "Image" column
                image_col = df['Image'].tolist()
                if col == 'Image':
                    for row_idx, image_path in enumerate(df['Image'], start=1):  # start=1 to skip header  
                        worksheet.write_url(row_idx, idx, image_path, string=image_path)
            # Save the Excel file
            writer.close()
            self.output_files.append(os.path.join(self.output, 'microstructure_analysis.xlsx'))
            
    def preprocess_image_for_ocr(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        return thresh
    
    # def ocr_with_easyocr(self, image: np.ndarray) -> str:
    #     """
    #     Perform OCR using easyocr to read the text from the image.
        
    #     :param image: Preprocessed image as a numpy array.
    #     :return: Extracted text as a string.
    #     """
    #     reader = easyocr.Reader(['en'], gpu=True)
    #     result = reader.readtext(image, detail=0)
    #     return " ".join(result)
    
    def extract_micrometer_value(self, ocr_result: str) -> str:
        pattern = r'(\d+\.?\d*)\s*um'
        match = re.search(pattern, ocr_result)
        if match:
            return match.group()
        else:
            return None
    
    def preprocess(self, img_path: str) -> None:
        '''Wrapper for all preprocessing methods.'''

        self.find_grain_boundaries(img_path, method='watershed')
        self.find_grain_boundaries(img_path, method='kmeans')

        #self.find_grain_boundaries(img_path, method='kmeans')
        #self.k_means_segmentation(img_path, 2)

    def show_grid(self, grid_view: list, title: str) -> None:
        '''
        Show grid of images.
        
        Args:
            grid_view: (list) - list of images to show
            title: (str) - title of the grid
        '''
        # Calculate the grid size based on the number of images
        grid_size = int(np.ceil(np.sqrt(len(grid_view))))
        fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))

        # Remove axes for empty subplots
        for i in range(grid_size*grid_size):
            if i >= len(grid_view):
                fig.delaxes(axs.flatten()[i])

        # Display images
        for i, grid_item in enumerate(grid_view):
            axs.flatten()[i].imshow(grid_item['img'], cmap=grid_item['cmap'])
            axs.flatten()[i].axis('off')

        plt.suptitle(title)
        plt.show()
        
    def kmeans_segmentation(self, img_path: str, n_clusters: int = 3) -> np.ndarray:
        '''
        Perform k-means clustering on the image.
        
        Args:
            img: (str) - image path to segment
            n_clusters: (int) - number of clusters
        '''
        #TODO when boundary is near closed shape (empty inside?), assign it to the same cluster
        
        create_if_not_exists(self.kmeans_output)
        img = cv2.imread(img_path)
        # Add k-means clustering here
        Z = img.reshape((-1,3)) 
        # convert to np.float32 
        Z = np.float32(Z) 
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) 
        ret, label, center = cv2.kmeans(Z, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS) 
        center = np.uint8(center) 
        res = center[label.flatten()] 
        res2 = res.reshape((img.shape)) 
        img_name, extension = os.path.splitext(os.path.basename(img_path))
        output_path = os.path.join(self.kmeans_output,  img_name + '_kmeans' + extension)
        self.save_to_output(output_path, res2)
        
    def generate_marker_colors(self, unique_markers):
        '''Generate colors for markers in watershed segmentation.'''
        # Generate a color map
        cmap = self.cmap
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # Create a new color map
        cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        marker_colors = {}
        for i, marker in enumerate(unique_markers):
            rgba_color = cmap(i % cmap.N)  # Use modulo to avoid index out of range
            # Convert to RGB format
            rgb_color = [int(255 * x) for x in rgba_color[:3]]
            if marker == -1:
                #grain boundaries
                marker_colors[marker] = [0, 255, 0]
            elif marker == 1:
                #martensite bands
                marker_colors[marker] = [0, 0, 255]
            else:
                #ferrite grains
                marker_colors[marker] = rgb_color
        return marker_colors
    
    def white_percentage(self, img: np.ndarray) -> float:
        return np.sum(img == 255) / img.size
    
    def grain_indicators(self, phase_mask: np.ndarray) -> dict:
        '''
        Calculate grain indicators of the phase mask.
        Args:
            phase_mask: (np.ndarray) - binary mask of the phase    
        '''
        grain_indicators = {}
        # Convert the binary mask to a 3-channel image
        phase_img = cv2.cvtColor(phase_mask*255, cv2.COLOR_GRAY2BGR)
        grains, _ = cv2.findContours(phase_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #draw contours on the image
        cv2.drawContours(phase_img, grains, -1, (0, 255, 0), 1)
        #cv2.imshow('Martensite bands', phase_img)
        for i, grain in enumerate(grains):
            x, y, w, h = cv2.boundingRect(grain)
            grain_perimeter = cv2.arcLength(grain, True)
            grain_area = cv2.contourArea(grain)
             
            rect_perimeter = 2 * (w + h)
            rect_area = w * h
            
            if grain_area < 0.01:
                continue
            edge_indicator = grain_perimeter / rect_perimeter
            shape_indicator = grain_perimeter / 4 * math.sqrt(grain_area)
            compactness_indicator = h * w / grain_area
            roundness_indicator = 4 * grain_area / (math.pi * (w**2 + h**2))
            
            grain_indicators[f'grain_{i}'] = {
                'edge_indicator': edge_indicator,
                'shape_indicator': shape_indicator,
                'compactness_indicator': compactness_indicator,
                'roundness_indicator': roundness_indicator,  
            }
        #cv2.imshow('Martensite bands', martensite_img)
        return grain_indicators
    
    def calculate_sharpness(self, img) -> float:
        '''Calculate the sharpness of the image.'''
        # Check if the image is already in grayscale
        if len(img.shape) == 3:
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            # If it is, just assign the image to img_grey
            img_grey = img
        # Calculate the Laplacian of the image
        lap = cv2.Laplacian(img_grey, cv2.CV_64F)
        return np.std(lap)
        
    def watershed_segmentation(self, img_path: str, comp_plot: bool = True) -> np.ndarray:
        '''
        Perform watershed segmentation on the image.
        Args:
            img_path: (str) - image path to segment 
            comp_plot: (bool) - if True, plot comparing original image and watershed image 
        '''
        create_if_not_exists(self.output)
        img = cv2.imread(img_path)
        i = 0
        while self.calculate_sharpness(img) > 100:
            #reduce noise while sharpness is > 100
            img = cv2.GaussianBlur(img, (3, 3), 0)
            i += 1
            
        img_grain_boundaries = img.copy()
        assert img is not None, f"Image not found at {img_path}."
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_threshold = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        if self.white_percentage(img_threshold) < self.martensite_threshold:
            # if white pixels (in default martensite bands) are > 60% of the image
            # switch black and white, because 
            # watershed algorithm works with white objects on black background
            img_threshold = cv2.bitwise_not(img_threshold)
        # noise removal
        kernel = np.ones((5,5), np.uint8)
        opening = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, kernel, iterations = 2)
        # sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_trans = cv2.distanceTransform(opening, cv2.DIST_L1, 3)
        _, sure_fg = cv2.threshold(dist_trans, 0.1*dist_trans.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1 
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        
        markers = cv2.watershed(img_rgb, markers)
        img_rgb[markers == -1] = [255, 0, 0]
        img_rgb[markers == 1] = [0, 0, 255]
        img_grain_boundaries[markers == -1] = [0, 255, 0]
    
        marker_colors = self.generate_marker_colors(np.unique(markers))
        for marker, color in marker_colors.items():
            img_rgb[markers == marker] = color
       
        
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        img_name, extension = os.path.splitext(os.path.basename(img_path))
        output_path = os.path.join(self.watershed_output,  img_name + '_watershed' + extension)
        output_path_boundaries = os.path.join(self.watershed_output,  img_name + '_boundaries' + extension)
        grid_view = [
            {'cmap': 'gray', 'img': img},
            {'cmap': 'gray', 'img': img_threshold},
            {'cmap': None, 'img': opening},
            {'cmap': None, 'img': dist_trans},
            {'cmap': None, 'img': sure_fg},
            {'cmap': None, 'img': img_rgb}
        ]
           
        if comp_plot:
            self.generate_comp_plot(img, img_rgb, markers, img_name, extension)
        self.save_to_output(output_path, img_rgb)
        #self.save_to_output(output_path_boundaries, img_grain_boundaries)
        
    def calculate_mechanical_properties(self, martensite_percentage: float, 
                                        ferrite_percentage: float) -> dict:
        mechanical_props = { 
            'mean_young_modulus': 
                    [round((MARTENSITE_PROPERITES['young_modulus'][0] * martensite_percentage + 
                    + FERRITE_PROPERITES['young_modulus'][0] * ferrite_percentage) / 100, 2), MARTENSITE_PROPERITES['young_modulus'][1]],
            'mean_yield_stress': 
                    [round((MARTENSITE_PROPERITES['yield_stress'][0] * martensite_percentage + 
                    + FERRITE_PROPERITES['yield_stress'][0] * ferrite_percentage) / 100, 2), FERRITE_PROPERITES['yield_stress'][1]],
            'mean_ultimate_tensile_strength': 
                    [round((MARTENSITE_PROPERITES['ultimate_tensile_strength'][0] * martensite_percentage + 
                    + FERRITE_PROPERITES['ultimate_tensile_strength'][0] * ferrite_percentage) / 100, 2), FERRITE_PROPERITES['ultimate_tensile_strength'][1]],
        }   
            
        return mechanical_props
    
    def save_mechanical_properties_record(self, plot_path: str, martensite_percentage: float,
                                          ferrite_percentage: float, mechanical_props: dict) -> None:
        if self.csv_raport is None:
            return None
        df = pd.read_csv(self.csv_raport)
        new_row = pd.DataFrame({
            'Image': [os.path.basename(plot_path)],
            'Martensite (%)': [round(martensite_percentage,2)],
            'Ferrite (%)': [round(ferrite_percentage,2)],
            'Boundaries (%)': [round(100 - martensite_percentage - ferrite_percentage,2)],
            'Mean E (GPa)': [mechanical_props['mean_young_modulus'][0]],
            'Mean σY (MPa)': [mechanical_props['mean_yield_stress'][0]],
            'Mean UTS (GPa)': [mechanical_props['mean_ultimate_tensile_strength'][0]]
        })
        new_row = new_row.dropna(how='all', axis=1)
        # Use concat to append the new row to the existing DataFrame
        #df = pd.concat([df, new_row], ignore_index=True)
        df = pd.concat([df.astype(new_row.dtypes), new_row.astype(df.dtypes)], ignore_index=True)
        df.to_csv(self.csv_raport, index=False)
    
    def generate_comp_plot(self, img: np.ndarray, img_rgb: np.ndarray,
                           markers: np.ndarray, img_name: str, extension: str) -> None:
        '''
        Generate comparison plot of original image and watershed image.
        Args:
            img: (np.ndarray) - original image
            img_rgb: (np.ndarray) - watershed image
            markers: (np.ndarray) - markers of the watershed image
            img_name: (str) - image name
            extension: (str) - image extension
        '''
        plt.rcParams['figure.figsize'] = [40, 10]
        plt.figure()
        #add figure title
        plt.suptitle(f'Martensite (white pixels) threshold: {self.martensite_threshold}',
                     fontsize=24)
        #make subplots bigger
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.title('Original Image')
        orig_subplot_patches=[
            mpatches.Patch(alpha=0, label=f'Sharpness - {round(self.calculate_sharpness(img),2)}')
        ]
        plt.legend(handles=orig_subplot_patches, bbox_to_anchor=(0.3,0.0),
                loc='upper right', prop={'size': 12})
        
        plt.subplot(122)
        plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), cmap='gray')
        plt.title('Watershed Image')
        # Add the legend to the plot
        num_pixels = len(markers.flatten())
        
        martensite_mask = np.uint8(markers == 1)
        boundaries_mask = np.uint8(markers == -1)
        ferrite_mask = np.uint8(markers > 1)
        
        martensite_grain_indicators = self.grain_indicators(martensite_mask)
        ferrite_grain_indicators = self.grain_indicators(ferrite_mask)
        
        martensite_percentage = np.sum(martensite_mask) / num_pixels * 100
        boundaries_percentage = np.sum(boundaries_mask) / num_pixels * 100
        ferrite_percentage = np.sum(ferrite_mask) / num_pixels * 100
        
        #generate table on plot
        #print('\n',ferrite_percentage+martensite_percentage+boundaries_percentage)
        
        mechanical_props = self.calculate_mechanical_properties(martensite_percentage, ferrite_percentage)
        
        grain_map_patches = [
            mpatches.Patch(color='green', label=f'Grain boundaries - {round(boundaries_percentage,2)}%'),
            mpatches.Patch(color='blue', label=f'Martensite bands - {round(martensite_percentage,2)}%'),
            mpatches.Patch(color=plt.get_cmap('RdGy')(0.2), label=f'Ferrite grains - {round(ferrite_percentage,2)}%'),
            mpatches.Patch(alpha=0, label=f"Mean E: {mechanical_props['mean_young_modulus'][0]} {mechanical_props['mean_young_modulus'][1]}"),
            mpatches.Patch(alpha=0, label=f"Mean σY: {mechanical_props['mean_yield_stress'][0]} {mechanical_props['mean_yield_stress'][1]}"),
            mpatches.Patch(alpha=0, label=f"Mean UTS: {mechanical_props['mean_ultimate_tensile_strength'][0]} {mechanical_props['mean_ultimate_tensile_strength'][1]}"),
        ]
        plt.legend(handles=grain_map_patches, bbox_to_anchor=(0.3,0.0), loc='upper right', prop={'size': 12})
        # Adjust layout to make room for the table:
        plt.subplots_adjust(left=0.2, bottom=0.2)
        plot_path = os.path.join(self.watershed_output,  img_name + '_compare' + extension)
        self.save_mechanical_properties_record(plot_path, martensite_percentage, ferrite_percentage, mechanical_props)
        plt.savefig(plot_path, dpi=200)
        self.output_files.append(plot_path)
        #plt.show()
        plt.close()
        
    def thresholding_segmentation(self, img_path: str) -> np.ndarray:
        '''
        Perform basic thresholding segmentation on the image.
        
        Args:
            img_path: (str) - image path to segment   
        '''
        create_if_not_exists(self.thresh_output)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_threshold = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    
        img_name, extension = os.path.splitext(os.path.basename(img_path))
        output_path = os.path.join(self.thresh_output,  img_name + '_threshold' + extension)
        self.save_to_output(output_path, img_threshold)
        
    def find_grain_boundaries(self, img_path: str, method: str) -> np.ndarray:
        '''Wrapper for all grain detection methods.'''
        available_methods = ['kmeans', 'thresholding', 'watershed']
        if method not in available_methods:
            raise ValueError(f"Method must be one of {available_methods}.")
        
        try:
            if method == 'kmeans':
                self.kmeans_output = os.path.join(self.output, 'kmeans')
                create_if_not_exists(self.kmeans_output)
                self.kmeans_segmentation(img_path)
            elif method == 'thresholding':
                self.thresh_output = os.path.join(self.output, 'thresholding')
                create_if_not_exists(self.thresh_output)
                self.thresholding_segmentation(img_path)
            elif method == 'watershed':
                self.watershed_output = os.path.join(self.output, 'watershed')
                create_if_not_exists(self.watershed_output)
                self.watershed_segmentation(img_path)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise e
            
    def save_to_output(self, filename: str, img: np.ndarray) -> None:
        '''Save img to self.output + filename path.'''
        path = os.path.join(self.output, filename)
        cv2.imwrite(path, img)
        self.output_files.append(path)
        
    def get_output_files(self) -> list:
        '''Returns the list of extracted images.'''
        self.run()
        #TODO return list of paths to the images (get it from DB)
        return self.output_files
    
    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        '''
        Parses only known command line arguments.
        Returns:
            argparse.Namespace: Parsed arguments.
        '''
        parser = argparse.ArgumentParser()
        parser.add_argument("-i", "--input", required=False, help="path to folder with images to segment")
        parser.add_argument("-o", "--output", required=False, help="path to output folder")
        parser.add_argument("--csv_raport", action='store_true', help="save mechanical properties to csv file")
        args, unknown = parser.parse_known_args()
        return argparse.Namespace(**vars(args))
    
def main():
    preprocessor = MicrostructureAnalyser()
       
    
    
if __name__ == '__main__':
    main()