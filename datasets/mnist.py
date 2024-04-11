from torchvision import transforms, datasets


def get_dataset(input_res: list[int]):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=input_res),
            transforms.Normalize(
                mean=[0.1307],
                std=[0.3081],
            ),
        ]
    )

    dataset = datasets.MNIST(
        root="./data",
        download=True,
        transform=transform,
    )

    return dataset
