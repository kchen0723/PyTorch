import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform_augment = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor()
])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_augment)

def show_images(dataset):
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    for i in range(5):
        image, label = dataset[i]
        axs[i].imshow(image.squeeze(0), cmap='gray')
        axs[i].set_title(f'label:{label}')
        axs[i].axis('off')
    plt.show()

show_images(dataset)