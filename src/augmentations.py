import albumentations as A
from albumentations.pytorch import ToTensorV2

# Augmentations
train_transform = A.Compose([
    A.OneOf([
        A.Blur(),
        A.MotionBlur(),
    ]),
    A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=10),
    A.OneOf([
        A.HorizontalFlip(),
        A.VerticalFlip(),
    ]),
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
])

valid_transform = A.Compose([
    A.Normalize(mean=(0.485,), std=(0.229,)),
    ToTensorV2(),
])
