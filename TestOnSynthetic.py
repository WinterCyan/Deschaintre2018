import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch import optim
from Model import *
from SaveLoad import *
from Dataset import *
from InNetworkRenderer import *
from Utils import *

test_dataset_path = '/media/winter/_hdd/MaterialDataset/Data_Deschaintre18/testBlended'
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
    test_data = MaterialDataset(test_dataset_path)
    test_dataloader = DataLoader(test_data, batch_size=1, pin_memory=True, shuffle=True)
    print("loaded {:d} material images, estimating SVBRDF...".format(len(test_data)))
    t1 = time.time()
    for i, sample in enumerate(test_dataloader):
        img_batch = sample["img"].to(device)  # [N,3,256,256]
        svbrdf_batch = sample["svbrdf"].to(device)  # [N,12,256,256]
        estimated_svbrdf_batch = model(img_batch)
        normals, diffuse, roughness, specular = expand_split_svbrdf(estimated_svbrdf_batch)
        normal_map = to_img(normals.squeeze(0).cpu())
        diffuse_map = to_img(diffuse.squeeze(0).cpu())
        roughness_map = to_img(roughness.squeeze(0).cpu())
        specular_map = to_img(specular.squeeze(0).cpu())
        gt_normals, gt_diffuse, gt_roughness, gt_specular = split_svbrdf(svbrdf_batch)
        gt_normal_map = to_img(gt_normals.squeeze(0).cpu())
        gt_diffuse_map = to_img(gt_diffuse.squeeze(0).cpu())
        gt_roughness_map = to_img(gt_roughness.squeeze(0).cpu())
        gt_specular_map = to_img(gt_specular.squeeze(0).cpu())
        rendered_img = to_img(renderer.render(generate_specular_scenes(1)[0], expand_svbrdf(estimated_svbrdf_batch)[0]).cpu())
        gt_img = to_img(renderer.render(generate_specular_scenes(4)[2], svbrdf_batch[0]).cpu())
        # gt_img = to_img(img_batch.squeeze(0).cpu())

        fig = plt.figure(figsize=(12, 5))
        # fig.tight_layout()
        plt.subplots_adjust(wspace=0.1,hspace=0.3)

        fig.add_subplot(2, 5, 1)
        plt.imshow(gt_img)
        plt.title("Rendered")
        plt.ylabel("Ground Truth")

        fig.add_subplot(2, 5, 2)
        plt.imshow(gt_normal_map)
        plt.title("Normal Map")
        plt.axis('off')

        fig.add_subplot(2, 5, 3)
        plt.imshow(gt_diffuse_map)
        plt.title("Albedo")
        plt.axis('off')

        fig.add_subplot(2, 5, 4)
        plt.imshow(gt_roughness_map)
        plt.title("Roughness")
        plt.axis('off')

        fig.add_subplot(2, 5, 5)
        plt.imshow(gt_specular_map)
        plt.title("Specular")
        plt.axis('off')

        fig.add_subplot(2, 5, 6)
        plt.imshow(rendered_img)
        plt.title("Rendered")
        plt.ylabel("Estimation")

        fig.add_subplot(2, 5, 7)
        plt.imshow(normal_map)
        plt.title("Normal Map")
        plt.axis('off')

        fig.add_subplot(2, 5, 8)
        plt.imshow(diffuse_map)
        plt.title("Albedo")
        plt.axis('off')

        fig.add_subplot(2, 5, 9)
        plt.imshow(roughness_map)
        plt.title("Roughness")
        plt.axis('off')

        fig.add_subplot(2, 5, 10)
        plt.imshow(specular_map)
        plt.title("Specular")
        plt.axis('off')

        # if i%(int(len(test_data)/10)) == 0:
        plt.savefig("estimation_results/synthetic/"+i.__str__()+".png")
        plt.close()
    t2 = time.time()
    t = t2 - t1
    print()
    print('-----------------------------')
    print('estimation done, results saved in folder "./estimation_results/synthetic"')
    print('estimation time: {:f}, average: {:3f}/img.'.format(t, t/len(test_data)))
