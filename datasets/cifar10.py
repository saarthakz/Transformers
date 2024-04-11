from torchvision import transforms, datasets


def get_dataset(input_res: list[int]):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(size=input_res),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2823, 0.2783, 0.2990],
            ),
        ]
    )

    dataset = datasets.CIFAR10(
        root="./data",
        download=True,
        transform=transform,
    )

    return dataset
