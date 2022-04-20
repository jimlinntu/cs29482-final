import torchvision
import numpy as np
from tqdm import tqdm

def main():
    train_dataset = torchvision.datasets.CIFAR10(root="./dataset/", train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="./dataset/", train=False, download=True)
    print("train length: {}".format(len(train_dataset)))
    print("test length: {}".format(len(test_dataset)))
    # flatten the pixel values
    with open("cifar_train.csv", "w") as f:
        for (img, label) in tqdm(train_dataset):
            npimg = np.array(img.resize((32, 32)))
            pixels = npimg.ravel().tolist()
            feature_str = ",".join(str(pixel) for pixel in pixels) + ",{}\n".format(label)
            f.write(feature_str)

    with open("cifar_test.csv", "w") as f:
        for (img, label) in tqdm(test_dataset):
            npimg = np.array(img.resize((32, 32)))
            pixels = npimg.ravel().tolist()
            feature_str = ",".join(str(pixel) for pixel in pixels) + ",{}\n".format(label)
            f.write(feature_str)




if __name__ == "__main__":
    main()
