"""
Module with utility functions for images
"""
import cv2
import numpy as np


def crop_bounding_box(full_image, extract_section, margin=.4, size=(64, 64)):
    """
    Crop a section from the image, add margin tot specified size
    :param full_image: full image containing the section to extract
    :param extract_section: area to extract (x top left, y top left, width, height)
    :param margin: add some margin to the section. Percentage of min(wifth, height)
    :param size: the result image resolution with be (width x height)
    :return: resized image in numpy array with shape (size x size x 3),
                bounding box of cropped image,
                cropped image,
    """
    img_height, img_width, _ = full_image.shape
    if extract_section is None:
        extract_section = [0, 0, img_width, img_height]
    (x, y, w, h) = extract_section
    margin_w = int(w * margin)
    margin_h = int(h * margin)
    margin_w = min(margin_w, x, (img_width - x - w))
    margin_h = min(margin_h, y, (img_height - y - h))
    x = x - margin_w
    y = y - margin_h
    w = w + 2 * margin_w
    h = h + 2 * margin_h
    cropped = full_image[y: y + h, x: x + w]
    resized_img = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img, (x, y, w, h), cropped


def draw_bounding_box_with_label(image, x, y, w, h, label):
    # size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 200, 0), 2)
    # cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def scale(array):
    """
    Scale values in x from [0..255] to [-1, 1]
    :param array: array to scale
    :return: rescaled array (float values)
    """
    array = array.astype('float32')
    array = array / 255.0
    array = array - 0.5
    array = array * 2.0
    return array


def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255):
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser
