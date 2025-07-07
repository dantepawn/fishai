### Functions to be used in the FishAi project
import cv2
from tqdm import tqdm
from pathlib import Path


def split_image(source_image_path , target_folder_path):
    """
    Splits an image into 4 equal parts and saves them in the target folder.
    """
    # Read the image
    image = cv2.imread(source_image_path)
    height, width = image.shape[:2]
    img_name = Path(source_image_path).name

    # check if the target folder exists 
    if not Path(target_folder_path).exists():
        print(f"Target folder {target_folder_path} does not exist. Creating it.")
    
    # Split the image into 4 equal parts
    subimage1 = image[:height//2,:width//2]
    subimage2 = image[:height//2,width//2:]
    subimage3 = image[height//2:,:width//2]
    subimage4 = image[height//2:,width//2:]

    for i, subimage in enumerate([subimage2, subimage1, subimage3, subimage4]):
        cv2.imwrite(target_folder_path+str(img_name).replace(".jpg", "_l" + str(i) + ".jpg"), subimage)
    


def create_yolo_labels(boxes, keypoints, target_folder="new_labels/"):
    """
    produces 1 file per image with the bounding boxes and keypoints in yolo format (normalized coordinates)
    
    boxes: dataframe with the bounding boxes
    keypoints: dataframe with the keypoints
    target_folder: folder where the labels will be saved
    """
    image_names = boxes['image_name'].unique()
    class_index  = 0 # fish

    for path in tqdm(image_names):

        # create a txt file for each image
        with open(target_folder+path.split(".")[0]+".txt", "w") as f:

            line = ""
            # bounding box
            for bb in boxes[boxes['image_name'] == path].to_dict('records'):
                original_height, original_width = bb['image_height'], bb['image_width']
                x_center = (bb['bbox_x'] + bb['bbox_width'] / 2) / original_width
                y_center = (bb['bbox_y'] + bb['bbox_height'] / 2) / original_height
                width =bb['bbox_width'] / original_width
                height = bb['bbox_height'] / original_width
                line += f"{class_index} {x_center} {y_center} {width} {height} "
                # keypoints only the keypoints that are inside the bounding box
                for i, row in keypoints[keypoints["filename"] ==path].iterrows():
                    if (row["x"] >= bb['bbox_x'] and row["x"] <= bb['bbox_x'] + bb['bbox_width']) and (row["y"] >= bb['bbox_y'] and row["y"] <= bb['bbox_y'] + bb['bbox_height']):
                        line += f"{row['x']/original_width} {row['y']/original_height} "
                line += "\n"
            f.write(line)

def remove_fins_morphological(mask, kernel_size=5, iterations=3):

    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask = (mask * 255).astype(np.uint8)

    # Create elliptical structuring element (better for biological shapes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Erode to disconnect fins from body
    eroded = cv2.erode(mask, kernel, iterations=iterations)

    # Dilate to restore body size while keeping fins removed
    dilated = cv2.dilate(eroded, kernel, iterations=iterations)

    # Find contours to isolate main body component
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Keep only largest contour (main fish body)
    largest_contour = max(contours, key=cv2.contourArea)
    cleaned_mask = np.zeros_like(mask)
    cv2.drawContours(cleaned_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return cleaned_mask
