from erasure.unlearners.torchunlearner import TorchUnlearner
from fractions import Fraction

import torch
import torch.nn.functional as F
from torch import nn

from erasure.core.factory_base import get_instance_kvargs

class Noise(nn.Module):
    """
    Noise class to add noise to the model weights. 
    """
    def __init__(self, batch_size, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

    def forward(self):
        return self.noise

class UNSIR(TorchUnlearner):
    def init(self):
        """
        Initializes the UNSIR class with global and local contexts.
        """
        super().init()

        self.epochs = self.local.config['parameters']['epochs'] 
        self.ref_data_retain = self.local.config['parameters']['ref_data_retain']
        self.ref_data_forget = self.local.config['parameters']['ref_data_forget']
        self.noise_lr = self.local.config['parameters']['noise_lr']

        self.predictor.optimizer = get_instance_kvargs(self.local_config['parameters']['optimizer']['class'],
                                      {'params':self.predictor.model.parameters(), **self.local_config['parameters']['optimizer']['parameters']})
        
        self.vocab_size = self.local.config['parameters']['vocab_size']

    
    def __unlearn__(self):
        """
        An implementation of the UNSIR unlearning algorithm proposed in the following paper:
        "Tarun, A.K., Chundawat, V.S., Mandal, M. and Kankanhalli, M., 2023. Fast yet effective machine unlearning. IEEE Transactions on Neural Networks and Learning Systems."
        Since the original method is thought for class-unlearning setting, we propose here the modified version proposed in the following paper: 
        "Choi, D. and Na, D., 2023. Towards machine unlearning benchmarks: Forgetting the personal identities in facial recognition systems. arXiv preprint arXiv:2311.02240."
        
        Codebase taken from this implementation: https://github.com/ndb796/MachineUnlearning
        """

        self.info(f'Starting UNSIR with {self.epochs} epochs for the impair phase')

        retain_loader, _ = self.dataset.get_loader_for(self.ref_data_retain, Fraction('0'))

        forget_loader, _ = self.dataset.get_loader_for(self.ref_data_forget, Fraction('0'))

        for epoch in range(self.epochs):
            running_loss = 0

            for batch_idx, ((x_retain, y_retain), (x_forget, y_forget)) in enumerate(zip(retain_loader, forget_loader)):
                y_retain = y_retain.to(self.device)
                batch_size_forget = y_forget.size(0)
                if self.vocab_size is None:
                    if x_retain.size(0) != retain_loader.batch_size or x_forget.size(0) != forget_loader.batch_size:
                        continue

                    noise_dim = x_forget.size()
                    noise = Noise(*noise_dim).to(self.device)
                    noise_optimizer = torch.optim.Adam(noise.parameters(), lr=self.noise_lr)
                    noise_tensor = noise()[:batch_size_forget]

                    # Update the noise for increasing the loss value.
                    for _ in range(5):
                        _, outputs = self.predictor.model(noise_tensor)
                        with torch.no_grad():
                            _, target_logits = self.predictor.model(x_forget.to(self.device))
                        # Maximize the similarity between noise data and forget features.
                        loss_noise = -F.mse_loss(outputs, target_logits)

                        # Backpropagate to update the noise.
                        noise_optimizer.zero_grad()
                        loss_noise.backward(retain_graph=True)
                        noise_optimizer.step()
                else:
                    # For text data, we need to create a noise tensor with the same shape as the input, it can't be optimized
                    # since the tokens are discrete, so we create a random tensor of the same shape.
                    seq_len = x_forget.size(2)

                    token_ids = torch.randint(
                            low=0,
                            high=self.vocab_size,
                            size=(batch_size_forget, seq_len),
                            device=self.device,
                            dtype=torch.long
                        )

                    attention_mask = torch.ones_like(token_ids, dtype=torch.long)
                    noise_tensor = torch.stack([token_ids, attention_mask], dim=1)
                    noise_tensor = noise_tensor.to(self.device)

                # Train the model with noise and retain image
                noise_tensor = torch.clamp(noise_tensor, 0, 1).detach().to(self.device)
                _, outputs = self.predictor.model(noise_tensor.to(self.device))
                loss_1 = self.predictor.loss_fn(outputs, y_retain)

                _, outputs = self.predictor.model(x_retain.to(self.device))
                loss_2 = self.predictor.loss_fn(outputs, y_retain)

                joint_loss = loss_1 + loss_2

                self.predictor.optimizer.zero_grad()
                joint_loss.backward()
                self.predictor.optimizer.step()
                running_loss += joint_loss.item() * x_retain.size(0)
            
            average_train_loss = running_loss / (len(retain_loader) * x_retain.size(0))
            
            self.info(f'UNSIR-1 - epoch = {epoch} ---> var_loss = {average_train_loss:.4f}')

        return self.predictor

    def check_configuration(self):
        super().check_configuration()

        self.local.config['parameters']['epochs'] = self.local.config['parameters'].get("epochs", 1)  # Default 1 epoch
        self.local.config['parameters']['ref_data_retain'] = self.local.config['parameters'].get("ref_data_retain", 'retain')  # Default reference data is retain
        self.local.config['parameters']['ref_data_forget'] = self.local.config['parameters'].get("ref_data_forget", 'forget')  # Default reference data is forget
        self.local.config['parameters']['noise_lr'] = self.local.config['parameters'].get("noise_lr", 0.01)  # Default noise learning rate is 0.01
        self.local.config['parameters']['optimizer'] = self.local.config['parameters'].get("optimizer", {'class':'torch.optim.Adam', 'parameters':{}})  # Default optimizer is Adam
        self.local.config['parameters']['vocab_size'] = getattr(self.global_ctx, 'vocab_size', None)