import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Функція для обчислення ентропії Шеннона для сегмента
def calculate_entropy(image_segment):
    pixel_values, counts = np.unique(image_segment, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# 2. Функція для обчислення середньоквадратичного відхилення для сегмента
def calculate_std_deviation(image_segment):
    return np.std(image_segment)

# 3. Функція для обчислення коефіцієнта кореляції для сегмента
def calculate_correlation(image_segment):
    mean = np.mean(image_segment)
    diff = image_segment - mean
    norm_diff = diff / np.std(image_segment)
    return np.mean(norm_diff[:-1] * norm_diff[1:])  # кореляція між сусідніми пікселями

# Функція для розбиття зображення на сегменти з урахуванням меж
def split_image(image, segment_size):
    h, w = image.shape
    segments = []
    for y in range(0, h, segment_size):
        for x in range(0, w, segment_size):
            segment = image[y:min(y + segment_size, h), x:min(x + segment_size, w)]
            segments.append(segment)
    return segments

# Функція для виділення сегментів, що перевищують поріг
def highlight_segments(image, segment_size, threshold, metric_map):
    h, w = image.shape[:2]
    highlighted_image = image.copy()

    for y in range(0, h, segment_size):
        for x in range(0, w, segment_size):
            segment_value = metric_map[y // segment_size, x // segment_size]

            # Якщо значення сегмента перевищує поріг, виділяємо його
            if segment_value > threshold:
                cv2.rectangle(highlighted_image, (x, y), (x + segment_size, y + segment_size), (0, 255, 0), 2)

    return highlighted_image

# Функція для побудови 3D гістограми
def plot_3d_histogram(data, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Створюємо координатну сітку
    x_data, y_data = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    z_data = np.zeros_like(x_data)
    dx = dy = 0.5
    dz = data.flatten()

    ax.bar3d(x_data, y_data, z_data, dx, dy, dz, shade=True)

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')

    plt.show()

# Основна функція для обробки зображення і виконання трьох завдань
def classify_image_segments(image_path):
    # Завантаження зображення
    image = cv2.imread(image_path)
    
    # Вивести оригінальне зображення
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Розміри сегментів
    segment_sizes = [8, 16, 32, 64, 128]

    # Пороги для кожного показника
    thresholds = {
        'entropy': 5.0,         # Вибраний поріг для ентропії
        'std_deviation': 20.0,  # Вибраний поріг для стандартного відхилення
        'correlation': 0.5      # Вибраний поріг для кореляції
    }

    # Спочатку виведемо зображення з виділеними сегментами
    for segment_size in segment_sizes:
        segments = split_image(gray_image, segment_size)

        entropies = [calculate_entropy(segment) for segment in segments]
        std_devs = [calculate_std_deviation(segment) for segment in segments]
        correlations = [calculate_correlation(segment) for segment in segments]

        # Формуємо мапи результатів
        entropy_map = np.array(entropies).reshape(gray_image.shape[0] // segment_size, gray_image.shape[1] // segment_size)
        std_map = np.array(std_devs).reshape(gray_image.shape[0] // segment_size, gray_image.shape[1] // segment_size)
        corr_map = np.array(correlations).reshape(gray_image.shape[0] // segment_size, gray_image.shape[1] // segment_size)

        # Виділення сегментів на основі порогів
        highlighted_entropy = highlight_segments(image, segment_size, thresholds['entropy'], entropy_map)
        highlighted_std = highlight_segments(image, segment_size, thresholds['std_deviation'], std_map)
        highlighted_corr = highlight_segments(image, segment_size, thresholds['correlation'], corr_map)

        # Виведення виділених сегментів
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(cv2.cvtColor(highlighted_entropy, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Entropy Highlighted {segment_size}x{segment_size}')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(highlighted_std, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'STD Deviation Highlighted {segment_size}x{segment_size}')
        axes[1].axis('off')

        axes[2].imshow(cv2.cvtColor(highlighted_corr, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Correlation Highlighted {segment_size}x{segment_size}')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    # Потім виведемо хітмапи
    for segment_size in segment_sizes:
        segments = split_image(gray_image, segment_size)

        entropies = [calculate_entropy(segment) for segment in segments]
        std_devs = [calculate_std_deviation(segment) for segment in segments]
        correlations = [calculate_correlation(segment) for segment in segments]

        # Формуємо мапи результатів
        entropy_map = np.array(entropies).reshape(gray_image.shape[0] // segment_size, gray_image.shape[1] // segment_size)
        std_map = np.array(std_devs).reshape(gray_image.shape[0] // segment_size, gray_image.shape[1] // segment_size)
        corr_map = np.array(correlations).reshape(gray_image.shape[0] // segment_size, gray_image.shape[1] // segment_size)

        # Виведення хітмапів
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(entropy_map, cmap='hot', interpolation='nearest')
        axes[0].set_title(f'Entropy Heatmap {segment_size}x{segment_size}')
        axes[0].axis('off')

        axes[1].imshow(std_map, cmap='hot', interpolation='nearest')
        axes[1].set_title(f'STD Deviation Heatmap {segment_size}x{segment_size}')
        axes[1].axis('off')

        axes[2].imshow(corr_map, cmap='hot', interpolation='nearest')
        axes[2].set_title(f'Correlation Heatmap {segment_size}x{segment_size}')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

        # Побудова 3D гістограм для кожної функції
        plot_3d_histogram(entropy_map, f'3D Entropy Histogram {segment_size}x{segment_size}')
        plot_3d_histogram(std_map, f'3D STD Deviation Histogram {segment_size}x{segment_size}')
        plot_3d_histogram(corr_map, f'3D Correlation Histogram {segment_size}x{segment_size}')

# Виклик функції для обробки вашого зображення
image_path = r'image\I22.BMP'
classify_image_segments(image_path)
