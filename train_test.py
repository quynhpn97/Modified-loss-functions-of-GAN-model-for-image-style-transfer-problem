import time
from options.train_option import TrainOptions
from options.test_option import TestOptions
from data import create_dataset
from model.dual_consistency_lossGAN import DualConsistencyGAN
import argparse
import sys
sys.argv=['']
del sys

import numpy as np
import matplotlib.pyplot as plt
def tensor2im(input_image, imtype=np.uint8):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

if __name__ == '__main__':
    opt = TrainOptions().initialize()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = DualConsistencyGAN(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
        print('End of epoch %d / %d \t' % (epoch, opt.n_epochs + opt.n_epochs_decay))

    # Test
    opt_test = TestOptions().initialize()
    dataset = create_dataset(opt_test)
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
    visuals = model.get_current_visuals()
    for i in visuals:
        img = visuals[i]
        print(i)
        img = tensor2im(img)
        img = transforms.ToPILImage(mode='RGB')(img.squeeze())
        plt.imshow(img)
        plt.show()
