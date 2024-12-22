import yaml
from train import train_gan
from evaluate import save_generated_trace

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    generator, min_vals, max_vals = train_gan(config)
    save_generated_trace(generator, config, min_vals, max_vals)

if __name__ == "__main__":
    main()
