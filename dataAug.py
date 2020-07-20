import imgaug.augmenters as iaa


seq = iaa.Sequential([
    iaa.Crop(px=(1, 30), keep_size=True),
    iaa.Fliplr(0.5),
    iaa.GaussianBlur(sigma=(0, 3.0))
])
seqDet = seq.to_deterministic()
def DoAugmentation(image):
    images_aug = seqDet(image=image)
    return images_aug