import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data

class AutoEncoder(nn.Module):
    def __init__(self):
        '''
            Model initialize
        '''
        super(AutoEncoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.pool = nn.MaxPool2d(2, return_indices = True)

        # Decoder
        self.unpool = nn.MaxUnpool2d(2)
        self.deconv2 = nn.ConvTranspose2d(16, 32, 3)
        self.deconv1 = nn.ConvTranspose2d(32, 3, 3)

        # Save for Unpooling
        self.ind = []
    
    def Encode(self, x):
        '''
            Encoder method
            2 cnn, 2 maxpooling
            Parameters:
                x: origin input tensor
            Return:
                x: output tensor after encoder
        '''
        x = torch.relu(self.conv1(x))
        x, ind1 = self.pool(x)
        x = torch.relu(self.conv2(x))
        x, ind2 = self.pool(x)
        self.ind = [ind1, ind2]
        return x
    
    def Decode(self, x):
        '''
            Decoder method
            2 cnn, 2 maxunpooling
            Parameters:
                x: input tensor
            Return:
                x: output tensor after decoder
        '''
        x = self.unpool(x, self.ind[1])
        x = torch.relu(self.deconv2(x))
        x = self.unpool(x, self.ind[0])
        x = torch.relu(self.deconv1(x))
        return x

class DataAugmentation(Data.Dataset):
    def __init__(self, data, label, transform = None):
        '''
            Augmentation Initialize
            -----------------------
            Parameters:
                data: numpy data of shape(numbers of img, 26, 26, 3)
                label: label of shape(numbers of img, 1)
                transform: use for transform numpy array
        '''
        self.data = data
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        '''
            Just a method implement getitem method
            --------------------------------------
            Parameters:
                idx: index of iterator
            Returns:
                item: the item after transform or not
                lable[idx]: label of that item
        '''
        item = self.data[idx]
        if self.transform: item = self.transform(item)
        return (item, label[idx])

def Training(device, data_loader, max_epoch = 50, draw = True):
    '''
        Training AutoEncoder
        --------------------
        Parameters:
            device: use cpu or gpu
            data_loader: pytorch DataLoader, need train data and label
            max_epoch: maximum training epoch
            draw: drawing training loss line or not
        Return:
            ae: after training, AutoEncoder model
    '''
    # Hyperparameter
    ae = AutoEncoder().double().to(device)
    evaluate = nn.MSELoss()
    optimizer = optim.AdamW(ae.parameters(), lr=0.001)

    plot_x = [i for i in range(max_epoch)]
    plot_training_loss = []

    # Training
    for epoch in range(max_epoch):
        total_loss, i = 0.0, 0
        for batch_x, batch_y in data_loader:
            # convert to gpu or cpu
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # clear gradient
            optimizer.zero_grad() 

            latent_code = ae.Encode(batch_x)
            outputs = ae.Decode(latent_code)
            loss = evaluate(outputs, batch_x)
            loss.backward()

            # update gradient
            optimizer.step() 

            i += 1
            total_loss += loss.item()
        print('epoch {}, loss: {:.4f}'.format(epoch, total_loss/i))
        plot_training_loss.append(total_loss/i)
    
    if draw:
        plt.plot(plot_x, plot_training_loss, lw=2, label='train_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.title('Training loss')
        plt.savefig('./Training')
        plt.show()
        plt.close()

    return ae

def Generater(model: AutoEncoder, input: torch.tensor, labal: np):
    '''
        Generater sample from origin data
        -------------------------------
        Parameters:
            model: Auto Encoder model
            input: origin dataset of shape(numbers of data, 3, 26, 26)
            label: input label shape of (numbers of data, 1)
    '''
    label_name = ['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None']
    # 9 class first index position 
    label_index = {414, 997, 405, 404, 415, 413, 412, 425, 0}

    gen_data = []
    gen_label = []

    for i, data in enumerate(input):
        # add axis to 4-dim
        data = data.unsqueeze(0)

        # Encoder, Decoder, Remove axis
        latent_code = model.Encode(data)
        org = model.Decode(latent_code)
        org = torch.squeeze(org, 0)

        # Add noise, Decode, Remove axis to 3-dim
        gen = []
        for _ in range(5):
            tmp = latent_code + torch.normal(mean = 0, std = 0.3, size = latent_code.size())
            tmp = torch.squeeze(model.Decode(tmp), 0)
            gen.append(tmp)
            gen_data.append(tmp.cpu().detach().numpy().transpose(2, 1, 0))
            gen_label.append(label[i])

        '''9 classes, generate img draw'''
        # if i in label_index:
        #     # Convert to numpy
        #     org_np = org.cpu().detach().numpy().transpose(2, 1, 0)
        #     gen_np = [gen[k].cpu().detach().numpy().transpose(2, 1, 0) for k in range(5)]
        #     label_index.remove(i)
        #     fig, ax = plt.subplots(1, 6)
        #     ax[0].imshow(np.argmax(org_np, 2))
        #     ax[0].set_title(label_name[label[i][0]], fontsize=10)
        #     for k in range(1, 6):
        #         ax[k].imshow(np.argmax(gen_np[k-1], 2))
        #         ax[k].set_title('gen'+str(k), fontsize=10)
        #     fig.savefig('./'+label_name[label[i][0]])
    
    # Save gen data and label
    gen_data = np.array(gen_data)
    gen_label = np.array(gen_label)

    np.save('./Data/gen_data', gen_data)
    np.save('./Data/gen_label', gen_label)
    print('generator save done !!!')

if __name__ == "__main__":
    
    # sad, computer is bad. so just use cpu.
    # "cuda:0" if torch.cuda.is_available() else 
    device = torch.device("cpu")

    # numpy array data
    data = np.load('./Data/data.npy')
    label = np.load('./Data/label.npy')

    # use for data augmentation
    '''
        Convert numpy to tensor
        And filp or rotate
    '''
    magic_effect = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-5, 5)),
    ])

    # Prepare Data Loader
    train_loader = Data.DataLoader(dataset=DataAugmentation(data, label, magic_effect), batch_size=4)

    # --- Start Training ---
    ae = Training(device, train_loader, max_epoch=20)

    # torch.save(ae, "./ae")
    # ae = torch.load('./ae')
    ae.eval()

    # Gen 1281*5
    data = torch.from_numpy(data.transpose(0, 3, 2, 1)).to(device)
    Generater(ae, data, label)

    '''
        local debug, see model is work well 
        Draw Graph for testing
    '''
    # org = np.load('./Data/data.npy')
    
    # pred = ae.Decode(ae.Encode(data)).cpu().detach().numpy().transpose(0, 3, 2, 1)
    # # show image. compare to origin
    # col = 20
    # for i in range(0, len(org), col):
    #     for idx in range(col):
    #         plt.subplot(2, col, idx+1)
    #         plt.imshow(np.argmax(org[i+idx], axis = 2))
    #         plt.subplot(2, col, idx+col+1)
    #         plt.imshow(np.argmax(pred[i+idx], axis = 2))
    #     plt.show()    