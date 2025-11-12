    
import torchvision.transforms as transforms
import torchvision
import os

def save_cifar10_real_images(num_images=60000):
        """保存CIFAR-10真实图像到目录"""
        print("保存CIFAR-10真实图像...")
        
        # 数据变换
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # 加载CIFAR-10数据集
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform
        )
        
        print(len(dataset))
        # 保存图像
        for i in range(min(num_images, len(dataset))):
            image, _ = dataset[i]
            
            # 转换为PIL图像并保存
            image_pil = transforms.ToPILImage()(image)
            image_pil.save(os.path.join("cifar_real_images", f'{i:05d}.png'))
        
        print(f"已保存 {min(num_images, len(dataset))} 张真实图像")


save_cifar10_real_images()