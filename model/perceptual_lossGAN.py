from base_gan import BaseGAN

from torchvision import transforms, models
vgg = models.vgg19(pretrained=True).features

for param in vgg.parameters():
    param.requires_grad_(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

def get_features(image, model, layers=None):

    if layers is None:
        layers = {'3': 'relu1_2',
                  '8': 'relu2_2',
                  '15': 'relu3_3', #content
                  '24': 'relu4_3'}

    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features

def gram_matrix(tensor):

    _, d, h, w = tensor.size()

    tensor = tensor.view(d, h * w)

    gram = torch.mm(tensor, tensor.t())

    return gram

class PerceptualGAN(BaseGAN):
    def __init__(self, opt):
        BaseGAN.__init__(self, opt)
        self.weight_content = opt.weight_content
        self.weight_style = opt.weight_style

    def set_input(opt):
        BaseGAN.set_input(self, opt)
        self.content_features = get_features(self.real_A, vgg)
        self.style_features = get_features(self.real_B, vgg)
        self.style_grams = {layer: gram_matrix(self.style_features[layer]) for layer in self.style_features}

    def backward_G(self):
        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # Loss Content
        target_features = get_features(self.fake_B, vgg)

        self.content_loss = torch.mean((target_features['relu3_3'] - self.content_features['relu3_3'])**2)

        # Loss Style
        self.style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            style_gram = self.style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram)**2)
            self.style_loss += layer_style_loss / (d * h * w)

        # Loss
        self.loss_G = self.loss_G_A +  self.weight_content*self.content_loss + self.weight_style*self.style_loss

        self.loss_G.backward()
