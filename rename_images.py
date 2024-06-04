import os

def rename_images(folder_path):
    # Get a list of all files in the directory
    files = os.listdir(folder_path)

    # Filter out non-image files (assuming images have .jpg, .jpeg, or .png extensions)
    image_extensions = ('.jpg', '.jpeg', '.png')
    images = [f for f in files if f.lower().endswith(image_extensions)]

    # Sort the images to ensure consistent naming order
    images.sort()

    # Rename each image
    for index, filename in enumerate(images):
        # Construct new file name
        new_name = f"img{index + 1}.jpg"
        
        # Get full path to the current file
        current_path = os.path.join(folder_path, filename)
        
        # Get full path to the new file name
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(current_path, new_path)

    print(f"Renamed {len(images)} images in {folder_path} successfully.")

# Specify the base path to your folders
base_path = r'C:\Users\User\Documents\GitHub\Tomato Leaf Disease Classification\train'

# List of folder names to iterate through
folder_names = ['Late_blight', 'Leaf_Mold', 'powdery_mildew', 'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_mosaic_virus', 'Tomato_Yellow_Leaf_Curl_Virus']

# Loop through each folder and call the rename_images function
for folder_name in folder_names:
    folder_path = os.path.join(base_path, folder_name)
    rename_images(folder_path)
