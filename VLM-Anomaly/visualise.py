import cv2
import matplotlib.pyplot as plt
import numpy as np
def visualize_anomaly(image_tensor, anomaly_map, shot):
    # Convert tensor to numpy array
    image_np = image_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
    image_np = (image_np * 255).astype(np.uint8)

    # Normalize the anomaly map for visualization
    anomaly_map = anomaly_map.squeeze()
    anomaly_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min())
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    # Apply a heatmap to the anomaly map
    heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Display the original image and the heatmap
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(heatmap, alpha=0.5)
    plt.title('Anomaly Heatmap '+shot)

    plt.show()