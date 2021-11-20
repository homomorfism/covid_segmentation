import csv
import sys
from datetime import datetime
from functools import reduce
from pathlib import Path

import numpy as np
from easydict import EasyDict
from tqdm import tqdm

from src.dataloader import CTDataLoader
from src.models import CTSemanticSegmentation

import pytorch_lightning as pl

from src.utils import rle_encoding

conf = EasyDict(
    dataset_folder=Path("data/preprocessed/"),
    batch_size=8,
    val_size=0.1
)


def join_arrays(data):
    converted = []
    for el in data:
        el = np.asarray(el, dtype=np.uint8)
        assert len(el.shape) == 3 and el.shape[1:] == (512, 512)

        el = np.transpose(el, axes=(1, 2, 0))
        assert el.shape[:2] == (512, 512)

        converted.append(el)

    return converted


def concat_images_with_same_label(outputs: list):
    names: list[str] = []
    frames: list[list] = []

    label: str
    previous = ""
    for label, mask in outputs:
        cleaned_label = label.rsplit("_", maxsplit=1)[0]

        if cleaned_label == previous:
            frames[-1].append(mask)

        else:
            names.append(cleaned_label)
            frames.append([mask])
            previous = cleaned_label

    return names, frames


def test(config: dict, model_path: Path, submit_path: Path):
    model = CTSemanticSegmentation.load_from_checkpoint(str(model_path), cfg=config)
    test_dataloader = CTDataLoader(train_aug=False).test_dataloader()
    trainer = pl.Trainer(gpus=0)

    outputs = trainer.predict(model, test_dataloader)
    masks = reduce(lambda x, y: x + y, (x['mask'] for x in outputs))
    labels = reduce(lambda x, y: x + y, (x['label'] for x in outputs))
    outputs = list(zip(labels, masks))

    names, frames = concat_images_with_same_label(outputs)
    frames = join_arrays(frames)

    with open(submit_path, "wt") as sb:
        submission_writer = csv.writer(sb, delimiter=",")

        submission_writer.writerow(["Id", "Predicted"])
        for name, frame in tqdm(zip(names, frames)):
            submission_writer.writerow([
                f"{name.split('.')[0]}",
                " ".join(rle_encoding(frame))
            ])

    print("finished!")


if __name__ == '__main__':
    model_path = Path("weights/patient_split")

    # Any params we want, does not influence the evaluation
    config = conf | {"loss_fn": "cross_entropy", "optim": "SGD", "lr": 0.001}

    date = datetime.now()
    test(config=config, model_path=model_path, submit_path=Path(f"kaggle_submits/submit-{date}.csv"))
