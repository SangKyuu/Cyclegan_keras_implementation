import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False
        self.initialize()

    def initialize(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--dataset_name', dest='dataset_name', default='horse2zebra', help='dataset name')
        parser.add_argument('--epochs', dest='epochs', type=int, default=200, help='# of epoch')
        parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr')
        parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
        parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
        parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--d_patch_size', dest='d_patch_size', type=int, default=16, help='discriminator out_dim')
        parser.add_argument('--data_pix_size', dest='data_pix_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--g_fir_dim', dest='g_fir_dim', type=int, default=32, help='# of gen filters in first conv layer')
        parser.add_argument('--d_fir_dim', dest='d_fir_dim', type=int, default=64, help='# of discri filters in first conv layer')
        parser.add_argument('--in_dim', dest='in_dim', type=int, default=3, help='# of input image channels')
        parser.add_argument('--out_dim', dest='out_dim', type=int, default=3, help='# of output image channels')
        parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--phase', dest='phase', default='train', help='train, test')
        parser.add_argument('--save_freq', dest='save_freq', type=int, default=1000, help='save a model every save_freq iterations')
        parser.add_argument('--sample_iter', dest='sample_iter', type=int, default=100, help='sampling iteration')
        parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
        parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
        parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
        parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
        parser.add_argument('--lambda_cycle', dest='lambda_cycle', type=float, default=10.0, help='cycle loss lambda')
        parser.add_argument('--lambda_id', dest='lambda_id', type=float, default=0.1, help='identity loss lambda')
        parser.add_argument('--max_size', dest='max_size', type=int, default=50, help='max size of image pool, 0 means do not use image pool')
        self.args, _ = parser.parse_known_args()
        self.initialized = True


p = BaseOptions()
opt = p.initialize()



