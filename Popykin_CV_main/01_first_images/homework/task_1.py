import cv2
import numpy as np
from pyparsing import deque


def find_way_from_maze(image: np.ndarray) -> tuple:
    """
    Найти путь через лабиринт.

    :param image: изображение лабиринта
    :return: координаты пути из лабиринта в виде (x, y), где x и y - это массивы координат
    """
     # Преобразуем изображение в градации серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

    height, width = binary_mask.shape

    # Находим начальную и конечную точки
    start_point = None
    end_point = None
    for i in range(height):
        for j in range(width):
            if binary_mask[i, j] == 0:
                start_point = (i, j)
                break
        if start_point is not None:
            break

    for i in range(height - 1, -1, -1):
        for j in range(width - 1, -1, -1):
            if binary_mask[i, j] == 0:
                end_point = (i, j)
                break
        if end_point is not None:
            break

    if start_point is None or end_point is None:
        print("Не удалось найти вход или выход")
        return None

    # Направления для движения (вверх, вниз, влево, вправо)
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Используем очередь для реализации алгоритма поиска
    queue = deque([start_point])
    visited_nodes = set()
    visited_nodes.add(start_point)
    # Словарь для хранения родительских узлов
    parent_map = {start_point: None}

    while queue:
        current_node = queue.popleft()

        # Если достигли конечной точки, выходим из цикла
        if current_node == end_point:
            break

        for move in movements:
            neighbor_node = (current_node[0] + move[0], current_node[1] + move[1])

            # Проверяем границы изображения и условия для добавления соседнего узла в очередь
            if (0 <= neighbor_node[0] < height and
                0 <= neighbor_node[1] < width and
                binary_mask[neighbor_node] == 0 and
                neighbor_node not in visited_nodes):

                visited_nodes.add(neighbor_node)
                queue.append(neighbor_node)
                parent_map[neighbor_node] = current_node

    # Восстанавливаем путь от конечной точки к начальной
    coords = []
    step = end_point

    while step is not None:
        coords.append(step)
        step = parent_map.get(step)

    coords.reverse()  # Переворачиваем список для получения правильного порядка координат
    x_coords, y_coords = ([], [])

    if coords:
        x_coords, y_coords = zip(*coords)  # Разделяем координаты на x и y

    return (x_coords, y_coords)

