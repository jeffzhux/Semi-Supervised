import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

class LongTailSSL(ImageFolder):
    def __init__(self,
            root: str,
            indexs: List,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
        ):
        super().__init__(
            root,
            transform,
            target_transform,
            loader,
            is_valid_file,
        )
        self.train = train
        self.imgs = np.array(self.imgs)[indexs]
        self.samples = np.array(self.samples)[indexs]
        self.targets = np.array(self.targets)[indexs]

        self.p_data = (np.unique(self.targets, return_counts=True)[1]/len(indexs)).tolist()
        self.cls_num_list = np.unique(self.targets, return_counts=True)[1]
    
    # def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
    #     """Finds the class folders in a dataset.

    #     See :class:`DatasetFolder` for details.
    #     """

    #     classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    #     class_and_num = []
    #     for entry in os.scandir(directory):
    #         if entry.is_dir():
    #             c = (entry.name, len(next(os.walk(f'{directory}/{entry.name}'))[2]))
    #             class_and_num.append(c)
    #     class_and_num = sorted(class_and_num, reverse=True, key=lambda x: x[1])

    #     if not class_and_num:
    #         raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    #     classes = [cls_name for cls_name, _ in class_and_num]
    #     class_to_idx = {cls_name: i for i, (cls_name, _) in enumerate(class_and_num)}

    #     return classes, class_to_idx
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = self.loader(img_path)

        if self.train:
            w, h = img.size
            pad = abs(w - h)
            if w > h:
                img = T.Pad((0, pad//2, 0, (pad-pad//2)), fill=(114, 114, 114))(img)
            elif h > w:
                img = T.Pad((pad//2, 0, (pad-pad//2), 0), fill=(114, 114, 114))(img)
        
        assert img.size[0] == img.size[1], f'image size is not equal, img size is {img.size}'

        
        if self.transform is not None:
            img = self.transform(img)

        target = int(target)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
    
class LongTailDataset(ImageFolder):
    def __init__(self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
        ):
        super().__init__(
            root,
            transform,
            target_transform,
            loader,
            is_valid_file,
        )
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """

        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        class_and_num = []
        for entry in os.scandir(directory):
            if entry.is_dir():
                c = (entry.name, len(next(os.walk(f'{directory}/{entry.name}'))[2]))
                class_and_num.append(c)
        class_and_num = sorted(class_and_num, reverse=True, key=lambda x: x[1])

        if not class_and_num:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        classes = [cls_name for cls_name, _ in class_and_num]
        class_to_idx = {cls_name: i for i, (cls_name, _) in enumerate(class_and_num)}

        return classes, class_to_idx
