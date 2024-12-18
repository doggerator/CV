import cv2
import numpy as np
import numpy as np
import math


def transform_coords(x, y, angle):
        rad_angle = math.radians(angle)
        new_x = x * math.cos(rad_angle) - y * math.sin(rad_angle)
        new_y = x * math.sin(rad_angle) + y * math.cos(rad_angle)
        return new_x, new_y

def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    angle = -angle

    height, width, _ = image.shape
    x_min = width
    x_max = 0
    y_min = height
    y_max = 0

    for y in range(height):
        for x in range(width):
            new_x, new_y = transform_coords(x, y, angle)

            if new_x < x_min:
                x_min = int(new_x)
            if new_x > x_max:
                x_max = int(new_x)

            if new_y < y_min:
                y_min = int(new_y)
            if new_y > y_max:
                y_max = int(new_y)

    new_height = y_max - y_min
    new_width = x_max - x_min
    rotated_image = np.zeros((new_height, new_width, 3), dtype=int)

    for y in range(height):
        for x in range(width):
            new_x, new_y = transform_coords(x, y)
            try:
                rotated_image[round(new_y - y_min), round(new_x - x_min)] = np.array(image[y, x], dtype=int)
            except IndexError:
                pass

    return rotated_image

import cv2
import numpy as np
def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    transformation_matrix = cv2.getAffineTransform(points1, points2)

    # Получаем размеры изображения
    img_height, img_width = image.shape[:2]

    # Определяем углы изображения
    corners = np.array([
        [0, 0],
        [0, img_height],
        [img_width, 0],
        [img_width, img_height]
    ], dtype=np.float32)

    # Применяем преобразование к углам
    ones = np.ones((4, 1), dtype=np.float32)
    corners_homogeneous = np.hstack((corners, ones))
    transformed_corners = np.dot(transformation_matrix, corners_homogeneous.T).T

    # Находим новые границы изображения
    x_min = np.min(transformed_corners[:, 0])
    x_max = np.max(transformed_corners[:, 0])
    y_min = np.min(transformed_corners[:, 1])
    y_max = np.max(transformed_corners[:, 1])

    # Корректируем матрицу преобразования для сдвига
    transformation_matrix[0, 2] -= x_min
    transformation_matrix[1, 2] -= y_min

    # Применяем аффинное преобразование к изображению
    new_width = round(x_max - x_min)
    new_height = round(y_max - y_min)
    transformed_image = cv2.warpAffine(image, transformation_matrix, (new_width, new_height))

    return transformed_image