from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from structured_data_train.gamma_util.prepared_image import LineImage, PreparedImage
from scipy import spatial

WHITE_PIXEL = np.array([255, 255, 255])


def split_image_by_color(input_image: np.ndarray) -> List[np.ndarray]:
    unique_colors = _extract_unique_colors(input_image)
    output_list = list()
    for color_index, color in enumerate(unique_colors):
        raw_im = (np.ones((320, 320, 3)) * 0).astype('uint8')
        mask = (
                (input_image[:, :, 0] == color[0]) *
                (input_image[:, :, 1] == color[1]) *
                (input_image[:, :, 2] == color[2])
        )
        raw_im[~mask] = WHITE_PIXEL
        output_list.append(raw_im)
    return output_list


def _extract_unique_colors(input_image: np.ndarray) -> List[List[int]]:
    all_colors = np.unique(input_image.reshape(320 * 320, 3), axis=0)
    all_colors = [list(c) for c in all_colors if (c != WHITE_PIXEL).all()]
    return sorted(all_colors)


def extract_orthogonal_lines_from_gray_image(
        input_image: np.ndarray,
        index: int
) -> LineImage:
    edges_arr = cv2.Canny(input_image, 50, 150)
    orthogonal_line_list = cv2.HoughLinesP(edges_arr, 1, np.pi / 2, 2, None, 10, 1)
    if orthogonal_line_list is None:
        orthogonal_line_list = list()
    # filtered_line_list = _remove_duplicate_lines(orthogonal_line_list)
    output_image = (np.zeros((320, 320, 3))).astype('uint8')
    filtered_line_list = np.array([line[0] for line in orthogonal_line_list])
    for line in filtered_line_list:
        x1, y1, x2, y2 = line
        cv2.line(output_image, (x1, y1), (x2, y2), (1, 1, 1), 1)
    original_line_image = LineImage(
        image_arr=output_image[:, :, 0],
        line_list=list(filtered_line_list),
    )
    return original_line_image


def _remove_duplicate_lines(
        orthogonal_line_list: List[np.ndarray],
        max_distance: int = 20*4,
) -> List[np.ndarray]:
    if len(orthogonal_line_list) == 0:
        return list()
    # orthogonal_line_arr = np.array([line[0] for line in orthogonal_line_list])
    orthogonal_line_arr = np.array([line for line in orthogonal_line_list])
    # orthogonal_line_arr = np.array([np.ndarray(line) for line in orthogonal_line_list])
    # print(orthogonal_line_arr.shape)
    tree = spatial.KDTree(orthogonal_line_arr)
    output_list = list()
    for current_index, current_line in enumerate(orthogonal_line_arr):
        neighbour_distance_list, neighbour_index_list = tree.query(current_line, k=2)
        if len(neighbour_index_list) == 1:
            output_list.append(current_line)
            continue

        distance = neighbour_distance_list[-1]
        neighbour_index = neighbour_index_list[-1]
        if distance > max_distance:
            output_list.append(current_line)
            continue
        if current_index < neighbour_index:
            left = orthogonal_line_arr[current_index]
            right = orthogonal_line_arr[neighbour_index]
            middle = ((left + right) / 2).astype(np.int32)
            # print(left, right, middle)
            # output_list.append(current_line)
            output_list.append(middle)

    final_output_list = list()
    for left_index, left_line in enumerate(output_list):
        found_smart = False
        for right_index, right_line in enumerate(output_list):
            if right_index == left_index:
                continue
            if is_horisontal(left_line) and is_horisontal(right_line):
                vertical_distance = abs(left_line[1] - right_line[1])
                if vertical_distance > 5:
                    continue
                new_line = [
                    min(left_line[0], right_line[0]),
                    (left_line[1] + right_line[1]) // 2,
                    max(left_line[2], right_line[2]),
                    (left_line[3] + right_line[3]) // 2,
                ]
                found_smart = True
                if right_index > left_index:
                    final_output_list.append(new_line)
            if is_vertical(left_line) and is_vertical(right_line):
                vertical_distance = abs(left_line[0] - right_line[0])
                if vertical_distance > 5:
                    continue
                new_line = [
                    (left_line[0] + right_line[0]) // 2,
                    max(left_line[1], right_line[1]),
                    (left_line[2] + right_line[2]) // 2,
                    min(left_line[3], right_line[3]),
                ]
                found_smart = True
                if right_index > left_index:
                    final_output_list.append(new_line)
        if not found_smart:
            final_output_list.append(left_line)

    x_of_vertical_lines = {line[0] for line in final_output_list if is_vertical(line)}
    if len(x_of_vertical_lines) > 7:
        print(f"Removed due to excess of vertical lines")
        return list()

    # print(f"Before: {len(output_list)}, after: {len(final_output_list)}")
    return final_output_list


def create_prepared_image(entry: pd.Series) -> PreparedImage:
    input_image = cv2.imread(entry['img_path'])
    sub_image_list = split_image_by_color(input_image)
    line_image_list = list()
    resized_line_image_list = list()
    for sub_image_index, sub_image in enumerate(sub_image_list):
        original_line_image = extract_orthogonal_lines_from_gray_image(sub_image, sub_image_index)
        line_image_list.append(original_line_image)
        # print(f"--- {sub_image_index}")
        resized_image = _resize_image(original_line_image)
        resized_line_image_list.append(resized_image)

    return PreparedImage(
        input_image,
        raw_sub_image_list=sub_image_list,
        line_image_list=line_image_list,
        resized_line_image_list=resized_line_image_list,
        image_type=entry.get('type'),
        label=entry.get('label'),
    )


def _resize_image(original_line_image: LineImage) -> LineImage:
    line_list = original_line_image.line_list
    if len(line_list) == 0:
        return original_line_image
    least_x = np.array([[line[0], line[2]] for line in line_list]).flatten().min()
    least_y = np.array([[line[1], line[3]] for line in line_list]).flatten().min()
    shifted_line_list = [
        [x1 - least_x + 3, y1 - least_y + 3, x2 - least_x + 3, y2 - least_y + 3]
        for x1, y1, x2, y2 in line_list
    ]
    big_x = np.array([[line[0], line[2]] for line in shifted_line_list]).flatten().max()
    big_y = np.array([[line[1], line[3]] for line in shifted_line_list]).flatten().max()

    image_downlosize_times = 1
    mult = min(308.0 / big_x, 308.0 / big_y) / image_downlosize_times

    resized_line_list = list()
    for line in shifted_line_list:
        resized_line = list()
        for e in line:
            resized_line.append(int(e * mult))
        resized_line_list.append(resized_line)
    filtered_line_list = _remove_duplicate_lines(resized_line_list)
    # print(f"Number of lines after removal of duplicates: {len(filtered_line_list)}")

    skip_drawing = False
    lens = [length_of_line(line) for line in filtered_line_list]
    if len(filtered_line_list) == 2:
        left, right = filtered_line_list
        if is_horisontal(left) and is_horisontal(right):
            skip_drawing = True
        if is_vertical(left) and is_vertical(right):
            skip_drawing = True
    if len(filtered_line_list) == 4:
        horizontal_lines = [line for line in filtered_line_list if is_horisontal(line)]
        vertical_lines = [line for line in filtered_line_list if is_vertical(line)]
        if len(horizontal_lines) == 2 and len(vertical_lines) == 2 and max(lens) < 50*4:
            skip_drawing = True
    if len(filtered_line_list) == 1:
        skip_drawing = True
    if len(lens) > 1 and max(lens) < 40*4:
        skip_drawing = True

    if len(filtered_line_list) == 0:
        second_resized_line_list = filtered_line_list
    else:
        # ==== Second resize start
        least_x = np.array([[line[0], line[2]] for line in filtered_line_list]).flatten().min()
        least_y = np.array([[line[1], line[3]] for line in filtered_line_list]).flatten().min()
        shifted_line_list = [
            [x1 - least_x + 1, y1 - least_y + 1, x2 - least_x + 1, y2 - least_y + 1]
            for x1, y1, x2, y2 in filtered_line_list
        ]
        big_x = np.array([[line[0], line[2]] for line in shifted_line_list]).flatten().max()
        big_y = np.array([[line[1], line[3]] for line in shifted_line_list]).flatten().max()
        mult = min(308.0 / big_x, 308.0 / big_y)
        second_resized_line_list = list()
        for line in shifted_line_list:
            resized_line = list()
            for e in line:
                resized_line.append(int(e * mult))
            second_resized_line_list.append(resized_line)
        # ==== Second resize end

    if len(filtered_line_list) == 0:
        third_resized_line_list = second_resized_line_list
    else:
        third_resized_line_list = list()
        smallest_x = min([line[0] for line in second_resized_line_list]+[line[2] for line in second_resized_line_list])
        biggest_x = max([line[0] for line in second_resized_line_list]+[line[2] for line in second_resized_line_list])
        smallest_y = min([line[1] for line in second_resized_line_list]+[line[3] for line in second_resized_line_list])
        biggest_y = max([line[1] for line in second_resized_line_list]+[line[3] for line in second_resized_line_list])
        for line in second_resized_line_list:
            if is_vertical(line):
                line = [
                    line[0],
                    smallest_y,
                    line[0],
                    biggest_y
                ]
            if is_horisontal(line):
                line = [
                    smallest_x,
                    line[1],
                    biggest_x,
                    line[1],
                ]
            third_resized_line_list.append(line)

    resized_image_arr = (np.zeros((320 // image_downlosize_times, 320 // image_downlosize_times, 3))).astype('uint8')
    if not skip_drawing:
        for line in third_resized_line_list:
            x1, y1, x2, y2 = line
            cv2.line(resized_image_arr, (x1, y1), (x2, y2), (1, 1, 1), 1)
    resized_image_arr = resized_image_arr[:, :, 1]
    resized_image = LineImage(
        image_arr=resized_image_arr,
        line_list=third_resized_line_list
    )
    return resized_image


def length_of_line(line):
    x1, y1, x2, y2 = line
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def is_horisontal(line):
    return line[1] == line[3]


def is_vertical(line):
    return line[0] == line[2]
