def normalizeRect(rect):
    rect = [*rect]
    if rect[2] < 0:
        rect[0], rect[2] = rect[0] + rect[2], -rect[2]
    if rect[3] < 0:
        rect[1], rect[3] = rect[1] + rect[3], -rect[3]
    return rect


def get_rect_intersection(rect1, rect2):
    left1, top1, width1, height1 = rect1
    left2, top2, width2, height2 = rect2

    left = max(left1, left2)
    top = max(top1, top2)
    right = min(left1 + width1, left2 + width2)
    bottom = min(top1 + height1, top2 + height2)

    # Check if there is an intersection
    if left < right and top < bottom:
        intersection_width = right - left
        intersection_height = bottom - top
        return [left, top, intersection_width, intersection_height]
    else:
        return None


def get_rect_area(rect):
    if not rect:
        return 0
    return abs(rect[2] * rect[3])


def formatTime(seconds):
    seconds = int(seconds)
    minutes = seconds // 60
    seconds %= 60
    return f"{minutes:02}:{seconds:02}"
