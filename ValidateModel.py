import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch import optim
from Model import *
from SaveLoad import *
from Dataset import *
from InNetworkRenderer import *
from Utils import *

val_dataset_path = '/media/winter/_hdd/MaterialDataset/Data_Deschaintre18/trainBlended'
device = 'cuda:0'
renderer = InNetworkRenderer()


if __name__ == '__main__':
    print('loading neural network...')
    model = MaterialNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    ckp_path = 'modelsavings/checkpoint_100epoch.pt'
    model, optimizer, load_epoch = load_ckp(ckp_path, model, optimizer)
    in_net_renderer = InNetworkRenderer()
    loss_func = MixLoss(renderer=in_net_renderer)

    model.eval()
    print('loading data...')
    data = MaterialDataset(data_dir=val_dataset_path)
    split_ratio = 0.02  # ratio for validation data
    _, validation_data = torch.utils.data.random_split(data, [int(math.ceil(len(data)*(1.0-split_ratio))), int(math.floor(len(data)*split_ratio))])
    validation_data_loader = DataLoader(validation_data, batch_size=8, pin_memory=True, shuffle=False)
    print("loaded {:d} material images, validating neural network...".format(len(validation_data)))
    val_loss = 0
    val_batch_num = 0
    for i, sample in enumerate(validation_data_loader):
        img_batch = sample["img"].to(device)  # [N,3,256,256]
        target_svbrdf_batch = sample["svbrdf"].to(device)  # [N,12,256,256]
        estimated_svbrdf_batch = model(img_batch)
        val_loss += loss_func(estimated_svbrdf_batch, target_svbrdf_batch).item()
        val_batch_num += 1
    val_loss /= val_batch_num
    print()
    print('----------------------------------------------')
    print("DONE.")
    print("Validation batch: {:d}, validation loss: {:.4f}".format(val_batch_num, val_loss))
    print("Similarity of SVBRDF: {:.2f}%.".format((1.0-val_loss)*100))
