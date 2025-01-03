###############################################################################
# 4. LSTM-BASED GENERATOR & DISCRIMINATOR (MULTI-STEP)
###############################################################################
class LSTMGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, seq_len, output_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_size=latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        lstm_out, (hn, cn) = self.lstm(z)  # (batch_size, seq_len, hidden_dim)
        out = self.fc_out(lstm_out)       # (batch_size, seq_len, 4)
        out = torch.tanh(out)             # [-1,1]
        return out

class LSTMDiscriminator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, 1)  # single logit

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)      # (batch_size, seq_len, hidden_dim)
        final_feature = lstm_out[:, -1, :]     # (batch_size, hidden_dim)
        logit = self.fc_out(final_feature)     # (batch_size, 1)
        return logit