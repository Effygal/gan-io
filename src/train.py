import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from models import LSTMGenerator, LSTMDiscriminator
from preprocess import preprocess_trace, denormalize_trace

def train_gan(config):
    data, min_vals, max_vals = preprocess_trace(config['data']['input_file'])
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float))
    loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

    G = LSTMGenerator(config['model']['latent_dim'], data.shape[1], config['model']['generator_hidden_dim']).cuda()
    D = LSTMDiscriminator(data.shape[1], config['model']['discriminator_hidden_dim']).cuda()

    optimizer_G = Adam(G.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['beta1'], 0.999))
    optimizer_D = Adam(D.parameters(), lr=config['training']['learning_rate'], betas=(config['training']['beta1'], 0.999))
    criterion = torch.nn.BCELoss()

    for epoch in range(config['training']['epochs']):
        for batch in loader:
            real_data = batch[0].cuda()
            bs = real_data.size(0)
            valid = torch.ones(bs, 1).cuda()
            fake = torch.zeros(bs, 1).cuda()

            z = torch.randn(bs, config['model']['sequence_length'], config['model']['latent_dim']).cuda()
            fake_data = G(z)

            optimizer_D.zero_grad()
            real_loss = criterion(D(real_data), valid)
            fake_loss = criterion(D(fake_data.detach()), fake)
            loss_D = real_loss + fake_loss
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            loss_G = criterion(D(fake_data), valid)
            loss_G.backward()
            optimizer_G.step()

    torch.save(G.state_dict(), "generator.pth")
    return G, min_vals, max_vals
