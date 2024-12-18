import cv2
import numpy as np


def find_road_number(image: np.ndarray) -> int:
    """
    Найти номер дороги, на которой нет препятсвия в конце пути.

    :param image: исходное изображение
    :return: номер дороги, на котором нет препятсвия на дороге
    """
    # Диапазоны для серого и красного цветов
    gray_lower = (200, 200, 200)
    gray_upper = (220, 220, 220)
    red_lower = (230, 0, 0)
    red_upper = (255, 50, 50)
    # Маски для этих цветов
    gray_mask = cv2.inRange(image, gray_lower, gray_upper)
    red_mask = cv2.inRange(image, red_lower, red_upper)
    # Находим контуры серых дорожек
    contours_gray, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Переворачиваем коллекцию (почему-то они записаны в обратном порядке...)
    contours_gray = contours_gray[::-1]
    road_number = None
    # Перебираем контуры серых дорожек
    for i, contour in enumerate(contours_gray):
        x, y, w, h = cv2.boundingRect(contour)
        # Проверяем наличие красного прямоугольника на этой дороге
        road_region = red_mask[y:y+h, x:x+w]
        if np.sum(road_region) == 0:
            road_number = i + 1
            break
    return road_number

