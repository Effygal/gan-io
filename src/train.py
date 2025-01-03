###############################################################################
# 5. TRAINING LOOP
###############################################################################
def train_gan(gen, disc, dataloader, device, latent_dim, seq_len,
              lrG, lrD, num_epochs, d_updates=1, g_updates=1):
    """
    Allows separate LR for G and D, plus user-specified d_updates/g_updates ratio.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizerG = optim.Adam(gen.parameters(), lr=lrG, betas=(0.5, 0.999))
    optimizerD = optim.Adam(disc.parameters(), lr=lrD, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        count_steps = 0

        for real_data in dataloader:
            real_data = real_data.to(device, dtype=torch.float)
            batch_size = real_data.size(0)

            # Prepare real/fake labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            #=============================
            # (1) Train Discriminator (d_updates times)
            #=============================
            d_loss_current = 0.0
            for _ in range(d_updates):
                optimizerD.zero_grad()

                # Real data forward
                d_out_real = disc(real_data)
                d_loss_real = criterion(d_out_real, real_labels)

                # Fake data forward
                z = torch.randn(batch_size, seq_len, latent_dim, device=device)
                fake_data = gen(z).detach()  # don't backprop through G
                d_out_fake = disc(fake_data)
                d_loss_fake = criterion(d_out_fake, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizerD.step()

                d_loss_current += d_loss.item()

            #=============================
            # (2) Train Generator (g_updates times)
            #=============================
            g_loss_current = 0.0
            for _ in range(g_updates):
                optimizerG.zero_grad()

                z = torch.randn(batch_size, seq_len, latent_dim, device=device)
                fake_data = gen(z)
                d_out_fake_for_g = disc(fake_data)
                g_loss = criterion(d_out_fake_for_g, real_labels)
                g_loss.backward()
                optimizerG.step()

                g_loss_current += g_loss.item()

            # Log average losses for this iteration
            d_loss_sum += d_loss_current / d_updates
            g_loss_sum += g_loss_current / g_updates
            count_steps += 1

        # Print epoch stats
        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"D Loss: {d_loss_sum/count_steps:.4f} | "
              f"G Loss: {g_loss_sum/count_steps:.4f}")
        
        # if epoch >= num_epochs // 2 and g_loss_sum/count_steps > 1.0:
        #     break

    return gen, disc, d_loss_sum/count_steps, g_loss_sum/count_steps