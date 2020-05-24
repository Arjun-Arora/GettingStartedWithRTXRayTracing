import matplotlib.pyplot as plt
import dataset
import numpy as np
from collections import OrderedDict


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

        np_img = np_img.squeeze()
        # if key == 'y_hat':
        #     np_img /= np.max(np_img)
        ax_arr_imgs[i].imshow(np.transpose(np_img, axes=(1, 2, 0)))
        ax_arr_imgs[i].title.set_text(key)

    if save:
        plt.savefig('viz_out.png')


def plot_loss(loss: float, iteration: int, save: bool):
    global line, ax_loss, iter_data, loss_data
    if line is None:
        fig_loss = plt.figure()
        ax_loss = fig_loss.add_subplot(111)
        ax_loss.set_yscale('log')
        line, = ax_loss.plot([iteration], [loss])
        plt.title('Loss')

    iter_data.append(iteration)
    loss_data.append(loss)

    ax_loss.relim()
    ax_loss.autoscale()

    line.set_xdata(iter_data)
    line.set_ydata(loss_data)
    plt.pause(0.05)

    if save:
        plt.savefig('loss.png')


f_imgs, ax_arr_imgs = None, None
line, ax_loss, iter_data, loss_data = None, None, [], []