import numpy as np
from skimage import color


def to_numpy(x):
    return x.detach().cpu().numpy()


# Utils for visualising

def visualize_prediction(image: np.array, prediction: np.array, true: np.array):
    image = (image - image.min()) / (image.max() - image.min())
    image = color.gray2rgb(image)
    new_image = image.copy()

    red, green = (1, 0, 0), (0, 1, 0)
    image[true == 1] = red
    new_image[prediction == 1] = green
    return np.concatenate([image, new_image], axis=1)


def rle_encoding(x: np.array):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return [str(item) for item in run_lengths]


def extract_image_names(frame_names: list[str]):
    cut_names = list(map(lambda x: x.rsplit("_", 1)[0], frame_names))
    return np.unique(cut_names)


def get_indices_of_frame_names(frame_names: list[str], selected_image_names: list[str]):
    boolean_mask = list(map(lambda name: name.rsplit("_", 1)[0] in selected_image_names, frame_names))
    indices = np.asarray(boolean_mask).nonzero()

    return indices[0]
