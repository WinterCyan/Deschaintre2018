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
    view_pos = generate_normalized_random_direction(count, min_eps=0.001, max_eps=0.1)
    light_pos = generate_normalized_random_direction(count, min_eps=0.001, max_eps=0.1)
    scenes = []
    for i in range(count):
        c = Camera(pos=view_pos[i])
        l = Light(pos=light_pos[i])
        scenes.append(Scene(c, l))

    return scenes


def generate_specular_scenes(count):
    view_pos = generate_normalized_random_direction(count, min_eps=0.001, max_eps=0.1)
    light_pos = view_pos * torch.Tensor([-1.0, -1.0, 1.0]).unsqueeze(0)
    distance_view = torch.exp(torch.Tensor(count, 1).normal_(mean=0.5, std=0.75))
    distance_light = torch.exp(torch.Tensor(count, 1).normal_(mean=0.5, std=0.75))
    shift = torch.cat([torch.Tensor(count, 2).uniform_(-1.0, 1.0), torch.zeros((count, 1))+0.0001], dim=-1)
    view_pos = view_pos * distance_view + shift
    light_pos = light_pos * distance_light + shift
    scenes = []
    for i in range(count):
        c = Camera(pos=view_pos[i])
        l = Light(pos=light_pos[i])
        scenes.append(Scene(c, l))

    return scenes



