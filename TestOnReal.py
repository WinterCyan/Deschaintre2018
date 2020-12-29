import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torchsummary import summary
import tensorwatch as watch
from torch import optim
from Model import *
from SaveLoad import *
from Dataset import *
from InNetworkRenderer import *
from Utils import *

dataset_path = '/media/winter/_hdd/MaterialDataset/Data_Deschaintre18/realCaptured'
device = 'cuda:0'
renderer = InNetworkRenderer()

if __name__ == '__main__':
    print('loading neural network...')
    model = MaterialNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    ckp_path = 'modelsavings/checkpoint_100epoch.pt'
    model, optimizer, load_epoch = load_ckp(ckp_path, model, optimizer)

    model.eval()
    print('loading data...')
    real_test_data = RealMaterialDataset(data_dir=dataset_path)
    test_dataloader = DataLoader(real_test_data, batch_size=1, pin_memory=True, shuffle=True)
    print("loaded {:d} material images, estimating SVBRDF...".format(len(real_test_data)))
    t1 = time.time()
    for i, sample in enumerate(test_dataloader):
        img_batch = sample["img"].to(device)  # [N,3,256,256]
        estimated_svbrdf_batch = model(img_batch)
        normals, diffuse, roughness, specular = expand_split_svbrdf(estimated_svbrdf_batch)
        normal_map = to_img(normals.squeeze(0).cpu())
        diffuse_map = to_img(diffuse.squeeze(0).cpu())
        roughness_map = to_img(roughness.squeeze(0).cpu())
        specular_map = to_img(specular.squeeze(0).cpu())
        rendered_img = to_img(renderer.render(generate_specular_scenes(1)[0], expand_svbrdf(estimated_svbrdf_batch)[0]).cpu())
        gt_img = to_img(img_batch.squeeze(0).cpu())

        fig = plt.figure(figsize=(12, 5))
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.01,hspace=0.01)

        fig.add_subplot(1, 6, 1)
        plt.imshow(gt_img)
        plt.axis('off')

        fig.add_subplot(1, 6, 2)
        plt.imshow(rendered_img)
        plt.axis('off')

        fig.add_subplot(1, 6, 3)
        plt.imshow(normal_map)
        plt.axis('off')

        fig.add_subplot(1, 6, 4)
        plt.imshow(diffuse_map)
        plt.axis('off')

        fig.add_subplot(1, 6, 5)
        plt.imshow(roughness_map)
        plt.axis('off')

        fig.add_subplot(1, 6, 6)
        plt.imshow(specular_map)
        plt.axis('off')

        plt.savefig('estimation_results/real/'+i.__str__()+".png")
        plt.close()
    t2 = time.time()
    t = t2 - t1
    print()
    print('-----------------------------')
    print('estimation done, results saved in folder "./estimation_results/real"')
    print('estimation time: {:.3f}s, average: {:.3f}s/img.'.format(t, t / len(real_test_data)))
    # plt.show()
