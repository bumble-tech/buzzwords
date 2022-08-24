from typing import List

import clip
import torch
from PIL import Image
from tqdm import tqdm


class CLIPEncoder():
    """
    Wrapper class for the CLIP encoder that can be slotted easily into
    Buzzwords for image topic modelling

    Parameters
    ----------
    model_name_or_path : str
        Name of model e.g. "ViT-B/32"

    Examples
    --------
    >>> embedding_model = CLIPEncoder('ViT-B/32')
    >>> paths = os.listdir('images')
    >>> embeddings = embedding_model.encode(paths)

    Encode set of images using `ViT-B/32`
    """
    def __init__(self, model_name_or_path: str, **kwargs):
        self.model, self.preprocess = clip.load(model_name_or_path, **kwargs)

    def encode(self,
               image_paths: List[str],
               batch_size: int = 8,
               show_progress_bar: bool = False) -> torch.Tensor:
        """
        Encode image paths into embeddings

        Parameters
        ----------
        image_paths : List[str]
            List of paths to images to be encoded
        batch_size : int
            Encode images in batches of this size
        show_progress_bar : bool
            Show progress bar when encoding

        Returns
        -------
        all_embeddings : torch.Tensor
            Embeddings of the images

        Notes
        -----
        This follows pretty much the format of the SentenceTransformers
        encode function, so it can be easily slotted in. Yes, ST does support
        image models - including CLIP - but we want the flexibility of the
        actual `clip` package so we use this instead
        """
        num_documents = len(image_paths)

        # Split remaining data into batches of size batch_size
        num_batches = (num_documents % batch_size > 0) + (num_documents // batch_size)

        all_embeddings = None

        for i in tqdm(range(num_batches), disable=not show_progress_bar):
            # Take sample of paths
            paths_sample = image_paths[i * batch_size: (i + 1) * batch_size]

            # Load and prep images for CLIP
            images = [self.preprocess(Image.open(path)).unsqueeze(0)
                      for path in paths_sample]

            with torch.no_grad():
                embeddings = self.model.encode_image(torch.vstack(images))

            # Store embeddings for output
            if all_embeddings is None:
                all_embeddings = embeddings
            else:
                all_embeddings = torch.vstack([
                    all_embeddings,
                    embeddings
                ])

        # Clear up memory, images are big bois
        del images
        del embeddings

        return all_embeddings
