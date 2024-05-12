from torchvision import transforms, datasets


def get_dataset(input_res: list[int], mean: list[float], std: list[float]):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(size=(178, 178)),
            transforms.Resize(size=input_res),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    dataset = datasets.CelebA(
        root="./data",
        download=True,
        transform=transform,
    )

    return dataset
