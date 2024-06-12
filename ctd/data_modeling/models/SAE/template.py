import pytorch_lightning as pl


class TemplateSAE(pl.LightningModule):
    def __init__(
        self,
        dataset: str,
    ):
        super().__init__()
        # Instantiate SAE model
        # To use fixed-point finding, must have a "decoder"
        # attribute with a "cell" attribute, where "cell" is a function
        # that takes input and hidden state and returns the new hidden state
        #

    def forward(self, data, inputs):
        # Pass data through the model
        # Inputs:
        # data: Tensor of shape (batch_size, seq_len, input_size)
        #           containing the spiking activity
        # inputs: Tensor of shape (batch_size, seq_len, input_size)
        #           containing the input to the model (if provided)
        #
        # Returns:
        # rates: Tensor of shape (batch_size, seq_len, input_size)
        #       containing the predicted log-firing rates (log if Poisson Loss is used)
        # latents: Tensor of shape (batch_size, seq_len, latent_size)
        #           containing the hidden state of the model
        # return rates, latents
        pass

    def configure_optimizers(self):
        # Define optimizer
        # Must return a pytorch optimizer
        pass

    def training_step(self, batch, batch_ix):
        # Define training step
        # Inputs:
        # batch: Tuple containing:
        #   - data: Tensor of shape (batch_size, seq_len, input_size)
        #           containing the spiking activity
        #   - data: (used if different IC encoding than recon activity)
        #   - inputs: Tensor of shape (batch_size, seq_len, input_size)
        #               containing the input to the model (if provided)
        #   - extra: Tuple containing any additional
        #           data needed for training (trial lens, etc.)

        # batch_ix: Index of the batch
        #
        # Returns:
        # loss: Tensor containing the loss for the batch
        pass

    def validation_step(self, batch, batch_ix):
        # Define validation step
        # Inputs:
        # batch: Tuple containing:
        #   - data: Tensor of shape (batch_size, seq_len, input_size)
        #           containing the spiking activity
        #   - data: (used if different IC encoding than recon activity)
        #   - inputs: Tensor of shape (batch_size, seq_len, input_size)
        #           containing the input to the model (if provided)
        #   - extra: Tuple containing any additional data
        #               needed for training (trial lens, etc.)

        # batch_ix: Index of the batch
        #
        # Returns:
        # loss: Tensor containing the loss for the batch
        pass
