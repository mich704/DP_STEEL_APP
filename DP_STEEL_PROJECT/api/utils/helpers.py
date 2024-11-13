import os
import sys
import shutil
import os
import logging

logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))


IMAGE_EXTENSIONS = (
    ".ras",
    ".xwd",
    ".bmp",
    ".jpe",
    ".jpg",
    ".jpeg",
    ".xpm",
    ".ief",
    ".pbm",
    ".tif",
    ".ppm",
    ".xbm",
    #".tiff",
    ".rgb",
    ".pgm",
    ".png",
    ".pnm",
)

RGB_COLORS = {
    "Yellow": [255, 255, 0],
    "Magenta": [255, 0, 255],
    "Cyan": [0, 255, 255],
    "Maroon": [128, 0, 0],
    "Dark Green": [0, 128, 0],
    "Navy": [0, 0, 128],
    "Olive": [128, 128, 0],
    "Purple": [128, 0, 128],
    "Teal": [0, 128, 128],
    "Orange": [255, 165, 0],
    "Pink": [255, 192, 203],
    "Brown": [165, 42, 42],
    "Gold": [255, 215, 0],
    "Silver": [192, 192, 192],
    "Violet": [238, 130, 238],
    "Indigo": [75, 0, 130],
    "Coral": [255, 127, 80],
    "Lime": [0, 255, 0]
}

def convert_windows_path(path: str) -> str:
    '''Converts windows path to unix path.'''
    return path.replace("\\", "/")

def create_if_not_exists(path: str, clear: bool = False) -> None:
    '''
    Creates a directory if it does not exist.

    Args:
        path (str): Path to the directory.
        clear (bool, optional): If True, removes all files and subdirectories in the directory. Defaults to False.
    '''
    if clear and os.path.exists(path):
        subfolders = [x for x in listdir_fullpath(path) if os.path.isdir(x)] 
        if len(subfolders) == 0:
            remove_files_in_path(path)
        else:
            for subfolder_path in subfolders:
                shutil.rmtree(subfolder_path)
    os.makedirs(path, exist_ok=True)


def remove_files_in_path(dir: str) -> None:
    '''Removes all files in a dir.'''
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        
def is_image(path: str) -> bool:
    '''Checks if the file is an image.'''
    return True if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS else False

def listdir_fullpath(dir: str):
    ''' Returns: list[str], list of paths to the files in the dir'''
    return [os.path.join(dir, f) for f in os.listdir(dir)]


def list_all_images(dir: str) -> list[str] | None:
    '''Returns: list[str] | None, list of paths to the images in the dir '''
    if os.path.isdir(dir):
        return [
            os.path.join(dir, f)
            for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f)) and is_image(f)
        ]
    else:
        return None

def progress_bar(iteration: int, total: int) -> None:
    '''
    Prints a progress bar to the console.
    
    Args:
        iteration: (int), current iteration
        total: (int), total number of iterations
    '''
    percent = "{:.1f}".format(100 * (iteration / float(total)))
    bar_length = 50
    filled_length = int(bar_length * iteration // total)
    bar = "=" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write(f"\rProgress: [{bar}] {percent}%")
    sys.stdout.flush()

def add_suffix_to_files(folder, suffix):
    '''Adds a suffix to all images in a folder.'''
    files = list_all_images(folder)
    for file in files:
        filename, extension = os.path.splitext(file)
        print(filename, suffix)
        if filename.endswith(suffix) == False:
            new_filename = f'{filename}{suffix}{extension}'
            print(new_filename)
            os.rename(file, new_filename)
            
def visualise_CNN(model_path: str, output_path: str = None) -> None:
    '''
    Visualises the model and saves it to the output path. 
    plot_model requires pydot and graphviz to be installed.
    
    Args:
        model_path: (str), path to the model
        output_path: (str), path to the folder where the visualisation will be saved
    '''
    import visualkeras
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from keras.models import load_model 
    from keras.utils import plot_model
    from PIL import ImageFont
    try:
        print(model_path)
        model = load_model(model_path)
        model_filename = os.path.splitext(os.path.basename(model_path))[0]
        if output_path is None:
            output_path = os.path.join(os.path.dirname(model_path), 'plots')
            print(output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        #add title to the plot
        font = ImageFont.truetype("arial.ttf", 32) 
        #3d layers view
        visualkeras.layered_view(model, legend=True, font=font,
                                to_file=os.path.join(output_path,f'{model_filename}_vis.png'))
        #layers description view
        plot_model(model,to_file=os.path.join(output_path,f'{model_filename}_desc.png'),
                   show_shapes=True, show_layer_names=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    