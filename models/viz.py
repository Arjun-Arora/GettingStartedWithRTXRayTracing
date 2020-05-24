import OpenEXR
import Imath
import matplotlib.pyplot as plt
import dataset
import numpy as np


def viz_exr(exr_filepath: str):
    img_tensor = dataset.exr_to_tensor(exr_filepath, False)
    np_arr = img_tensor.numpy()
    np_arr = np.transpose(np_arr, (2, 1, 0))
    print(np.min(np_arr), np.max(np_arr))
    plt.imshow(np_arr)
    plt.show()


viz_exr('data/Clean-0.exr')
