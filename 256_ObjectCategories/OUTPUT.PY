import matplotlib.pyplot as plt
import numpy as np

def show_sample_images(dataloader, model):
    model.eval()
    images, labels = next(iter(dataloader))
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, 5, figsize=(15,3))
    for idx in range(5):
        img = np.transpose(images[idx].numpy(), (1, 2, 0))
        axes[idx].imshow(img)
        axes[idx].set_title(f'Pred: {preds[idx]}, Actual: {labels[idx]}')
    plt.show()

show_sample_images(test_loader, trained_model)

