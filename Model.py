from ModelParts import *

ngf = 64
input_channel = 3
output_channel = 9


class MaterialNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = InitEncoder(c_in=input_channel, c_out=ngf, c_gb_out=2*ngf)
        self.encoder2 = Encoder(c_in=ngf, c_out=2*ngf, c_gb_in=2*ngf, c_gb_out=4*ngf)
        self.encoder3 = Encoder(c_in=2*ngf, c_out=4*ngf, c_gb_in=4*ngf, c_gb_out=8*ngf)
        self.encoder4 = Encoder(c_in=4*ngf, c_out=8*ngf, c_gb_in=8*ngf, c_gb_out=8*ngf)
        self.encoder5 = Encoder(c_in=8*ngf, c_out=8*ngf, c_gb_in=8*ngf, c_gb_out=8*ngf)
        self.encoder6 = Encoder(c_in=8*ngf, c_out=8*ngf, c_gb_in=8*ngf, c_gb_out=8*ngf)
        self.encoder7 = Encoder(c_in=8*ngf, c_out=8*ngf, c_gb_in=8*ngf, c_gb_out=8*ngf)
        self.encoder8 = LastEncoder(c_in=8*ngf, c_out=8*ngf, c_gb_in=8*ngf, c_gb_out=8*ngf)

        self.decoder8 = InitDecoder(c_in=8*ngf, c_out=8*ngf, c_gb_in=8*ngf, c_gb_out=8*ngf)
        self.decoder7 = Decoder(c_in=8*ngf, c_link=8*ngf, c_out=8*ngf, c_gb_in=8*ngf, c_gb_out=8*ngf)
        self.decoder6 = Decoder(c_in=8*ngf, c_link=8*ngf, c_out=8*ngf, c_gb_in=8*ngf, c_gb_out=8*ngf)
        self.decoder5 = Decoder(c_in=8*ngf, c_link=8*ngf, c_out=8*ngf, c_gb_in=8*ngf, c_gb_out=8*ngf)
        self.decoder4 = Decoder(c_in=8*ngf, c_link=8*ngf, c_out=4*ngf, c_gb_in=8*ngf, c_gb_out=4*ngf)
        self.decoder3 = Decoder(c_in=4*ngf, c_link=4*ngf, c_out=2*ngf, c_gb_in=4*ngf, c_gb_out=2*ngf)
        self.decoder2 = Decoder(c_in=2*ngf, c_link=2*ngf, c_out=ngf, c_gb_in=2*ngf, c_gb_out=ngf)
        self.decoder1 = LastDecoder(c_in=ngf, c_link=ngf, c_out=output_channel, c_gb_in=ngf)

    def forward(self, batch_input):
        en1, gb1 = self.encoder1(batch_input)
        en2, gb2 = self.encoder2(en1, gb1)
        en3, gb3 = self.encoder3(en2, gb2)
        en4, gb4 = self.encoder4(en3, gb3)
        en5, gb5 = self.encoder5(en4, gb4)
        en6, gb6 = self.encoder6(en5, gb5)
        en7, gb7 = self.encoder7(en6, gb6)
        en8, gb8 = self.encoder8(en7, gb7)

        de8, gb9 = self.decoder8(en8, gb8, dropout=0.5)
        de7, gb10 = self.decoder7(de8, en7, gb9, dropout=0.5)
        de6, gb11 = self.decoder6(de7, en6, gb10, dropout=0.5)
        de5, gb12 = self.decoder5(de6, en5, gb11)
        de4, gb13 = self.decoder4(de5, en4, gb12)
        de3, gb14 = self.decoder3(de4, en3, gb13)
        de2, gb15 = self.decoder2(de3, en2, gb14)
        de1 = self.decoder1(de2, en1, gb15)

        return de1


# train pipeline:
# input pair: [N,3*5,H,W], light pos, view pos
# [N,0:3,H,W] -> PredictionNet -> [N,9,H,W](+lightpos, viewpos) -> Renderer -> [N,3,H,W]
#                   |<------- loss --<---|---------<---------------------------------|

# input_img: [N, 3*5, H, W] + lightpos + viewpos
# output_params: [N, 9, H, W], normal + albedo +
def render_loss(input_params, lightpos, viewpos, output_params):
    N, C_total, H, W = input_params.shape
    C = C_total / 5.0
    GT_rendered = input_params[:, 0:C, :, :]
    GT_normal = input_params[:, C:2 * C, :, :]
    GT_albedo = input_params[:, 2 * C:3 * C, :, :]
    GT_roughness = input_params[:, 3 * C:4 * C, :, :]
    GT_specular = input_params[:, 4 * C:5 * C, :, :]
    extended_params = extend_output(output_params)  # [N,3*4,H,W], normal + albedo + roughness + specular
    output_normal = extended_params[:, 0:C, :, :]
    output_albedo = extended_params[:, C:2 * C, :, :]
    output_roughness = extended_params[:, 2 * C:3 * C, :, :]
    output_specular = extended_params[:, 3 * C:4 * C, :, :]
    output_rendered = torch_renderer(
        normal=output_normal,
        albedo=output_albedo,
        roughness=output_roughness,
        specular=output_specular,
        lightpos=lightpos,
        viewpos=viewpos
    )
    loss = img_l2_loss(GT_rendered, output_rendered)  # confirm that img are > 0, < 1 ???
    return loss


def extend_output(params):
    extended = params
    return extended


def torch_renderer(normal, albedo, roughness, specular, lightpos, viewpos):
    rendering = normal
    return rendering


def img_l2_loss(img1, img2):
    diff = torch.log(img1+0.01) - torch.log(img2+0.01)
    loss = torch.sum(diff**2)
    return loss
