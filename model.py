'''
code based on https://github.com/xhujoy/CycleGAN-tensorflow/blob/master/model.py
'''
from __future__ import division
import os
import numpy as np
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt


from data_loader import DataLoader
from module import *
from utils import *
from option import *



class cyclegan(object):
    def __init__(self, opt):
        self.discriminator = discriminator
        self.generator = generator_resnet
        self.criterionGAN = mae_criterion
        self.opt = opt
        self.img_shape = (self.opt.data_pix_size, self.opt.data_pix_size, self.opt.in_dim)
        self.build_model()
        self.pool = ImagePool(self.opt.max_size)

    def build_model(self):
        self.lambda_id = self.opt.lambda_id * self.opt.lambda_cycle
        self.d_A = discriminator(self.opt)
        self.d_B = discriminator(self.opt)
        self.optimizer = Adam(self.opt.lr, self.opt.beta1)
        self.d_A.compile(loss='mse',
                         optimizer=self.optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=self.optimizer,
                         metrics=['accuracy'])
        self.g_AB = generator_resnet(self.opt)
        self.g_BA = generator_resnet(self.opt)
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        #style transfer
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        #reconstructure
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        #identity
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        #to train generator
        self.d_A.trainable = False
        self.d_B.trainable = False

        #dicriminator determine validity
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discrgit/model.py:67iminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])

        self.combined.compile(loss=['mean_squared_error', 'mean_squared_error',
                                    'mean_absolute_error', 'mean_absolute_error',
                                    'mean_absolute_error', 'mean_absolute_error'],
                              loss_weights=[1, 1,
                                            self.opt.lambda_cycle, self.opt.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=self.optimizer)




    def train(self):
        start_time = datetime.datetime.now()
        # Calculate output shape of D (PatchGAN)
        patch = int(self.opt.d_patch_size)
        self.disc_patch = (patch, patch, 1)
        # Adversarial loss ground truths
        valid = np.ones((self.opt.batch_size,) + self.disc_patch)
        fake = np.zeros((self.opt.batch_size,) + self.disc_patch)
        self.data_loader = DataLoader(dataset_name=self.opt.dataset_name,
                                      img_res=(self.opt.data_pix_size, self.opt.data_pix_size))
        for epoch in range(self.opt.epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(self.opt.batch_size)):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict_on_batch(imgs_A)
                fake_A = self.g_BA.predict_on_batch(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)


                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                        [valid, valid,
                                                        imgs_A, imgs_B,
                                                        imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                                                                        % ( epoch, self.opt.epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss[0], 100*d_loss[1],
                                                                            g_loss[0],
                                                                            np.mean(g_loss[1:3]),
                                                                            np.mean(g_loss[3:5]),
                                                                            np.mean(g_loss[5:6]),
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % self.opt.sample_iter == 0:
                    self.sample_images(epoch, batch_i)
        model_json = self.combined.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.combined.save_weights("model.h5")
        print("Saved model to disk")




    def sample_images(self,epoch, batch_i):
        os.makedirs('images/%s' % self.opt.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.opt.dataset_name, epoch, batch_i))
        plt.close()



'''def test(self, opt):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if opt.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif opt.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(opt.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # write html for visual comparison
        index_path = os.path.join(opt.test_dir, '{0}_index.html'.format(opt.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.testB, self.test_A) if opt.which_direction == 'AtoB' else (
            self.testA, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, opt.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(opt.test_dir,
                                      '{0}_{1}'.format(opt.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
'''

if __name__ == '__main__':
    p = BaseOptions()
    opt = p.args
    gan = cyclegan(opt)
    gan.train()