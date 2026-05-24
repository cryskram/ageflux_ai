import torch.nn as nn

adversarial_loss = nn.MSELoss()

classification_loss = nn.CrossEntropyLoss()

reconstruction_loss = nn.L1Loss()
