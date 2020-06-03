import matplotlib.pyplot as plt
import dataset
import numpy as np
from collections import OrderedDict

GAMMA = 2.2
INV_GAMMA = 1 / GAMMA


def viz_exr(exr_filepath: str):
    img_tensor = dataset.exr_to_tensor(exr_filepath, False)
    np_arr = img_tensor.numpy()
    print(np_arr.shape)
    np_arr = np.transpose(np_arr, (1, 2, 0))
    print(np.min(np_arr), np.max(np_arr))
    plt.imshow(np_arr)
    plt.show()


def visualize_outputs(to_plot: OrderedDict, epoch: int, batch_i: int, save: bool):
    global f_imgs, ax_arr_imgs
    if ax_arr_imgs is None:
        f_imgs, ax_arr_imgs = plt.subplots(1, len(to_plot.items()))

    f_imgs.suptitle('Epoch: {epoch}, Batch: {batch}'.format(epoch=epoch, batch=batch_i))
    for i, (key, value) in enumerate(to_plot.items()):
        np_img = value.numpy()
        if np_img.shape[0] > 1:
            np_img = np_img[0]

        np_img = (np_img.squeeze() * 255).astype(np.uint8)
        # if key == 'y_hat':
        #     np_img /= np.max(np_img)
        print(key, "->", "min:", np.min(np_img), "max:", np.max(np_img))
        ax_arr_imgs[i].imshow(np.transpose(np_img, axes=(1, 2, 0)))
        ax_arr_imgs[i].title.set_text(key)

    if save:
        plt.savefig('viz_out.png')


def tensor_preprocess(img_tensors, downsample_factor=4, difference=False):
    np_img = img_tensors.numpy()[:, ::downsample_factor, ::downsample_factor]
    if difference:
        np_img = np.linalg.norm(np_img, axis=0)

    return np_img


def plot_loss(loss: float, iteration: int, save: bool, val: bool):
    global line1, line2, ax_loss, train_iter_data, train_loss_data, val_iter_data, val_loss_data
    if line1 is None and line2 is None:
        fig_loss = plt.figure()
        ax_loss = fig_loss.add_subplot(111)
        ax_loss.set_yscale('log')
        line1, line2, = ax_loss.plot([iteration], [loss], [], [])
        plt.title('Loss')

    ax_loss.relim()
    ax_loss.autoscale()

    if val:
        val_iter_data.append(iteration)
        val_loss_data.append(loss)

        line2.set_xdata(val_iter_data)
        line2.set_ydata(val_loss_data)
    else:
        train_iter_data.append(iteration)
        train_loss_data.append(loss)

        line1.set_xdata(train_iter_data)
        line1.set_ydata(train_loss_data)

    plt.pause(0.05)

    if save:
        plt.savefig('loss.png')


f_imgs, ax_arr_imgs = None, None
line1, line2, ax_loss = None, None, None
train_iter_data, train_loss_data, val_iter_data, val_loss_data = [], [], [], []
