from InNetworkRenderer import *
import Utils
from Utils import *


class Camera:
    def __init__(self, pos):
        self.pos = pos


class Light:
    def __init__(self, pos):
        self.pos = pos


class Scene:
    def __init__(self, camera, light):
        self.camera = camera
        self.light = light


def generate_random_scenes(count):
    view_pos = generate_normalized_random_direction(count)
    light_pos = generate_normalized_random_direction(count)
    scenes = []
    for i in range(count):
        c = Camera(pos=view_pos[i])
        l = Light(pos=light_pos[i])
        scenes.append(Scene(c, l))

    return scenes


def generate_specular_scenes(count):
    view_pos = generate_normalized_random_direction(count)
    light_pos = view_pos * torch.Tensor([-1.0, -1.0, 1.0]).unsqueeze(0)
    distance_view = Utils.generate_distance()
    distance_light = Utils.generate_distance()
    shift = torch.cat([torch.Tensor(count, 2).uniform_(-0.5, 0.5), torch.zeros((count, 1))+0.0001], dim=-1)
    view_pos = view_pos * distance_view + shift
    light_pos = light_pos * distance_light + shift
    scenes = []
    for i in range(count):
        c = Camera(pos=view_pos[i])
        l = Light(pos=light_pos[i])
        # c = Camera(torch.Tensor([-0.5141, -1.6318,  2.0391]))
        # l = Light(torch.Tensor([0.4125, 1.2524, 4.0322]))
        scenes.append(Scene(c, l))

    return scenes


