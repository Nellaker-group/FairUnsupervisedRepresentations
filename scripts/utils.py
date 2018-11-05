import numpy as np
import PIL

###

class CellDataset(Dataset):
    def __init__(self, numpy_array_file, class_labels_array_file, transformations):
        self.data = np.load(numpy_array_file)
        self.classes = np.load(class_labels_array_file)
        self.transformations = transformations
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        image_array = self.data[idx, :, :, :]
        image_class = self.classes[idx]
        image = PIL.Image.fromarray(image_array)
        image = self.transformations(image)
        return {"image": image, "class": image_class}
