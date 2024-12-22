from preprocess import denormalize_trace

def evaluate_mmd(real_data, synthetic_data):
    return np.mean((real_data.mean(axis=0) - synthetic_data.mean(axis=0)) ** 2)

def save_generated_trace(generator, config, min_vals, max_vals):
    z = torch.randn(1000, config['model']['sequence_length'], config['model']['latent_dim']).cuda()
    synthetic_data = generator(z).cpu().detach().numpy()
    denorm_data = denormalize_trace(synthetic_data, min_vals, max_vals)
    with open(config['data']['output_file'], 'w') as f:
        for row in denorm_data:
            f.write(f"{row[0]:.6f} {'R' if row[1] < 0.5 else 'W'} {int(row[2])} {int(row[3])}\n")
