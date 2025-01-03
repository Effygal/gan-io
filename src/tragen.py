###############################################################################
# 6. GENERATION + INVERSE-TRANSFORM (PER COLUMN)
###############################################################################
def generate_synthetic(gen, scalers, output_path, device, latent_dim, seq_len, num_entries):
    ts_scaler, length_scaler, lba_scaler, lat_scaler = scalers
    gen.eval()

    gen_bs = 64
    all_fakes = []
    rows_generated = 0

    with torch.no_grad():
        while rows_generated < num_entries:
            current_bs = min(gen_bs, (num_entries - rows_generated) // seq_len + 1)
            if current_bs <= 0:
                break

            z = torch.randn(current_bs, seq_len, latent_dim, device=device)
            fake_seq = gen(z).cpu().numpy()  # (current_bs, seq_len, 4)

            # Flatten for inverse scaling
            fake_seq_2d = fake_seq.reshape(-1, 4)

            # columns
            fake_ts   = fake_seq_2d[:, [0]]
            fake_len  = fake_seq_2d[:, [1]]
            fake_lba  = fake_seq_2d[:, [2]]
            fake_lat  = fake_seq_2d[:, [3]]

            ts_unscaled   = ts_scaler.inverse_transform(fake_ts)
            len_unscaled  = length_scaler.inverse_transform(fake_len)
            lba_unscaled  = lba_scaler.inverse_transform(fake_lba)
            lat_unscaled  = lat_scaler.inverse_transform(fake_lat)

            combined_unscaled = np.hstack([ts_unscaled, len_unscaled, lba_unscaled, lat_unscaled])
            all_fakes.append(combined_unscaled)
            rows_generated += current_bs * seq_len

    all_fakes = np.concatenate(all_fakes, axis=0)
    all_fakes = all_fakes[:num_entries]

    # Write to file
    with open(output_path, 'w') as f:
        for row in all_fakes:
            out_str = ' '.join(str(int(x)) for x in row)
            f.write(out_str + '\n')

    print(f"Saved synthetic data to {output_path} (total lines = {len(all_fakes)})")