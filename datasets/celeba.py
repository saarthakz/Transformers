from torchvision import transforms, datasets


def get_dataset(input_res: list[int]):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(size=(178, 178)),
            transforms.Resize(size=input_res),
            transforms.Normalize(
                mean=[0.5084, 0.4224, 0.3767],
                std=[0.3012, 0.2788, 0.2773],
            ),
        ]
    )

    dataset = datasets.CelebA(
        root="./data",
        download=True,
        transform=transform,
    )

    return dataset
