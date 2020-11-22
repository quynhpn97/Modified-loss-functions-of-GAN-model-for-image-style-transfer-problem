'''
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
'''
from option import BaseOptions

class TestOptions(BaseOptions):

    def initialize(self):
        parser = BaseOptions().initialize()
        parser.add_argument('--path_A', type=str, default=['drive/My Drive/Python/CycleGAN/TestContent/*'], help='path for content images')
        parser.add_argument('--path_B', type=str, default="drive/My Drive/Python/CycleGAN/Style/style2.jpg", help='path for style images')
        return parser.parse_args()
