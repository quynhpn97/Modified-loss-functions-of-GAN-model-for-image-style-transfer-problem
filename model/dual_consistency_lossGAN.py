from base_gan import BaseGAN

from torchvision import transforms, models
vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

def get_features(image, model, layers=None):

    if layers is None:
        layers = {'11': 'relu3_1'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

class DualConsistencyGAN(BaseGAN):
    def __init__(self, opt):
        BaseGAN.__init__(self, opt)
        self.weight_content = opt.weight_content
        self.weight_style = opt.weight_style

    def set_input(opt):
        BaseGAN.set_input(self, opt)
        self.content_features = get_features(self.real_A, vgg)
        self.style_features = get_features(self.real_B, vgg)

    def backward_G(self):
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # Loss Content
        target_features = get_features(self.fake_B, vgg)

        self.content_loss = torch.mean((target_features['relu3_1'] - self.content_features['relu3_1'])**2)
        # Loss Style
        layers = {'9': 'relu'
                  }

        features_fake = {}
        E_fakeB_ = self.fake_B
        features_style = {}
        E_styleB_ = self.real_B
        for name, layer in self.netG_A.module.model._modules.items():
            E_fakeB_ = layer(E_fakeB_)
            E_styleB_ = layer(E_styleB_)
            if name in layers:
                features_fake[layers[name]] = E_fakeB_
                features_style[layers[name]] = E_styleB_
        self.style_loss = torch.mean((features_fake['relu'] - features_style['relu'])**2)

        # Loss
        self.loss_G = self.loss_G_A +  self.weight_content*self.content_loss + self.weight_style*self.style_loss

        self.loss_G.backward()
