from ModelParts import *
import torch.nn as nn
from Utils import *
from Environment import *

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


class L1Loss(nn.Module):
    def forward(self, input_batch, target_batch):
        # input_batch: [N, 9, H, W]
        # target: [N, 12, H, W]
        estimated_normals, estimated_diffuse, estimated_roughness, estimated_specular = expand_split_svbrdf(input_batch)
        target_normals, target_diffuse, target_roughness, target_specular = expand_split_svbrdf(target_batch)
        estimated_diffuse = torch.log(estimated_diffuse+0.01)
        estimated_specular = torch.log(estimated_specular+0.01)
        target_diffuse = torch.log(target_diffuse+0.01)
        target_specular = torch.log(target_specular+0.01)

        return nn.functional.l1_loss(estimated_normals, target_normals) + nn.functional.l1_loss(estimated_diffuse, target_diffuse) + nn.functional.l1_loss(estimated_roughness, target_roughness) + nn.functional.l1_loss(estimated_specular, target_specular)


class RenderingLoss(nn.Module):
    def __init__(self, renderer):
        super(RenderingLoss, self).__init__()
        self.renderer = renderer
        self.random_scenes_count = 3
        self.specular_scenes_count = 6

    def forward(self, input_batch, target_batch):
        # input_batch: [N, 9, H, W]
        # target: [N, 12, H, W]
        input_svbrdf = expand_svbrdf(input_batch)  # [N, 12, H, W]
        batch_size = input_batch.shape[0]
        estimated_renderings_batch = []
        target_renderings_batch = []
        for i in range(batch_size):
            scenes = generate_random_scenes(count=self.random_scenes_count) + generate_specular_scenes(self.specular_scenes_count)

            estimated_svbrdf = input_svbrdf[i]  # [12,H,W]
            target_svbrdf = target_batch[i]
            estimated_renderings = []
            target_renderings = []

            for scene in scenes:
                estimated_renderings.append(self.renderer.render(scene, estimated_svbrdf))
                target_renderings.append(self.renderer.render(scene, target_svbrdf))
            estimated_renderings_batch.append(torch.cat(estimated_renderings, dim=0))
            target_renderings_batch.append(torch.cat(target_renderings, dim=0))

        estimated_renderings_batch_log = torch.log(torch.stack(estimated_renderings_batch, dim=0)+0.1)
        target_renderings_batch_log = torch.log(torch.stack(target_renderings_batch, dim=0)+0.1)

        loss = nn.functional.l1_loss(estimated_renderings_batch_log, target_renderings_batch_log)
        return loss


class MixLoss(nn.Module):
    def __init__(self, renderer, l1_weight=0.1):
        super(MixLoss, self).__init__()
        self.l1_weight = l1_weight
        self.l1_loss = L1Loss()
        self.rendering_loss = RenderingLoss(renderer=renderer)

    def forward(self, input_batch, target_batch):
        return self.l1_weight*self.l1_loss(input_batch, target_batch) + self.rendering_loss(input_batch, target_batch)
