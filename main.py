import torch
from torch.utils.data import DataLoader
from model import ImagenTrainer

BATCH_SIZE = 32
trainer = ImagenTrainer(csv_file="./wojak_data.csv", 
                        root_dir="./processed_memes", 
                        epochs=20000, 
                        img_size=64)

data = ImagenTrainer.load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
