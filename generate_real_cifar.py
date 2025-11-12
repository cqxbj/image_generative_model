    
import torchvision.transforms as transforms
import torchvision
import os

def save_cifar10_real_images(num_images=60000):
  
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        
        print(len(dataset))
    
        for i in range(min(num_images, len(dataset))):
            image, _ = dataset[i]
            image_pil = transforms.ToPILImage()(image)
            image_pil.save(os.path.join("cifar_real_images", f'{i:05d}.png'))
def save_cifar10_test_images():
      pass

save_cifar10_real_images()