import cv2
import os
from typing import List, Dict, Tuple

def group_duplicate_images(folder: str) -> List[List[str]]:
    image_groups: Dict[Tuple, List[str]] = {}

    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            abs_path = os.path.abspath(path)

            img = cv2.imread(abs_path)
            if img is None:
                continue

            key = (img.shape, img.dtype.str, img.tobytes())

            if key not in image_groups:
                image_groups[key] = []
            image_groups[key].append(abs_path)

    result = []
    for group in image_groups.values():
        if len(group) > 1:
            result.append(group)

    return result


if __name__ == '__main__':
    groups = group_duplicate_images('/home/project/dataset')
    for g in groups:
        print(g)