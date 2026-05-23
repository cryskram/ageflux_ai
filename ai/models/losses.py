import torch.nn as nn

adversarial_loss = nn.BCELoss()
reconstruction_loss = nn.L1Loss()
