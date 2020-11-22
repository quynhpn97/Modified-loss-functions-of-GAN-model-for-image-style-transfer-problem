'''
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
'''
from option import BaseOptions

class TrainOptions(BaseOptions):

    def initialize(self):
        parser = BaseOptions().initialize()
        # training parameters
        parser.add_argument('--path_A', type=str, default=['drive/My Drive/Python/CycleGAN/Content/*'], help='path for content images')
        parser.add_argument('--path_B', type=str, default="drive/My Drive/Python/CycleGAN/Style/style2.jpg", help='path for style images')
        parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--isTrain', default = True)
        parser.add_argument('--weight_content', type=float, default=1.0, help='weight for content loss')
        parser.add_argument('--weight_style', type=float, default=0.1, help='weight for style loss')
        return parser.parse_args()
