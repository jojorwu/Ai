"""
Vision Encoder to process images for multimodal input.
"""
from backend import np
from nn_components.linear import Linear


class VisionEncoder:
    """
    A simple Vision Encoder that converts an image into a sequence of embeddings.
    It uses a single convolutional layer to create patches and project them.
    """
    def __init__(self, d_model: int, patch_size: int, num_channels: int):
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection = Linear(patch_size * patch_size * num_channels, d_model)

    def forward(self, images: np.ndarray) -> np.ndarray:
        """
        Processes a batch of images into patch embeddings.
        Args:
            images (np.ndarray): A batch of images with shape (B, H, W, C).
        Returns:
            np.ndarray: A sequence of embeddings for each image with shape (B, num_patches, d_model).
        """
        batch_size, height, width, _ = images.shape
        p = self.patch_size
        num_patches_h = height // p
        num_patches_w = width // p

        patches = images.reshape(batch_size, num_patches_h, p, num_patches_w, p, self.num_channels)
        patches = patches.transpose(0, 1, 3, 2, 4, 5)
        patches = patches.reshape(batch_size, num_patches_h * num_patches_w, -1)

        patch_embeddings = self.projection.forward(patches)
        return patch_embeddings

    def get_children(self):
        """Returns a dictionary of child layers."""
        return {'projection': self.projection}

    def get_trainable_params(self):
        """Returns all trainable parameters of this layer."""
        return self.projection.get_trainable_params()
