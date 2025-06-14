import torch


# Для каждой матрицы brc[i] обнулить строку row_idx и столбец col_idx, указанные в row_indices[i] и col_indices[i] соответственно
@torch.compile
def zerify_r_c_batched(brc, row_indices, col_indices):
    b, n, m = brc.shape

    batch_indices = torch.arange(b)

    brc[batch_indices, row_indices, :] = -1  # -1, а не 0 т.к. мы так сначала будем идти по 0 из фиктивных. иначе
    brc[batch_indices, :, col_indices] = -1  # рискуем заново посчитать уже посчитнанное и занулить строки/столбцы с 1

    return brc


# расчёт маски для score (b, n, m)
# real_target_mask - маска (b, m) не фиктивных target значений (target значения дополняются 0 для обработки в батч-режиме)
@torch.compile
def align(_score, real_target_mask):
    # _score = score.detach()  # нет нужды считать градиент, здесь вычисляется маска для значений output - target

    # изменение лосса на обратно пропорциональное для более удобного обращения:
    # некорректные/уже учтёные значения можно просто занулить, и это будет считаться как самое плохое значение из возможных
    _score = 1 + 1 / (torch.abs(_score) + 1)

    # применение маски, учитывающий, какие из target значений не пустышки (пустышкам даётся самое плохое значение - 0)
    _score = _score * real_target_mask

    b_mask = torch.arange(_score.shape[0])  # маска для батчей
    mask = torch.zeros_like(_score)  # будущий результат, маска для score

    # проходим поэтапно по всем максимальным значениям (в батч-режиме)
    for i in range(min(_score.shape[-2:])):  # неизместно, кого меньше: output или target

        # ищем индекс наилучшего значения (в батч-режиме, т.е. ищем свой индекс лучшего значения в каждом батче)
        flatten_max_id = torch.argmax(torch.flatten(_score, start_dim=1), dim=1)
        # преобразуем в удобный вид
        max_cords = torch.unravel_index(flatten_max_id, _score.shape[1:])
        # координаты лучшей пары output - target (батч-режим)
        b_min_cord_o, b_min_cord_t = max_cords

        # real_target_mask[b_mask, 0, i] - нужно для того, чтобы не писать фиктивные target'ы, когда настоящие закончились
        # (цикл то всё равно продолжается до конца, а у маленьких масок real_target_mask все значения станут нулями,
        # когда реальные таргеты закончатся и max будет выдавать случайный ноль)
        mask[b_mask, b_min_cord_o, b_min_cord_t] = real_target_mask[b_mask, 0, i]  # записываем в маску

        _score = zerify_r_c_batched(_score, b_min_cord_o, b_min_cord_t)  # зануляем уже учтённое

    return mask


# лосс между output (b, n, cl + 4) и target (b, m, cl + 4). нотация bbox: "cxcywh"
# считается, что формат output всегда одинаков, а все значения target дополнены 0 до идентичной размерности (при необходимости)
@torch.compile
def loss_v2(output, target, num_classes: int, _eps=1e-3, threshold_iou_for_class_loss=0.75,
            k_bbox_loss=1.0, k_class_loss=1.0):
    output_bbox, output_class = output.split((4, num_classes), 2)
    target_bbox, target_class = target.split((4, num_classes), 2)

    # (batch_size, n, 1) - (batch_size, 1, m) = (batch_size, n, m) (с помощью трансляции вычисляем попарные разности)
    x_diff = output_bbox[:, :, 0].unsqueeze(2) - target_bbox[:, :, 0].unsqueeze(1)
    y_diff = output_bbox[:, :, 1].unsqueeze(2) - target_bbox[:, :, 1].unsqueeze(1)

    # расстояние между всеми парами для всех комбинаций output - target
    distance_diff = torch.sqrt(x_diff ** 2 + y_diff ** 2)

    # tl и br углы
    o_tl_x = (output_bbox[:, :, 0] - output_bbox[:, :, 2] / 2).unsqueeze(2)
    o_tl_y = (output_bbox[:, :, 1] - output_bbox[:, :, 3] / 2).unsqueeze(2)
    o_br_x = (output_bbox[:, :, 0] + output_bbox[:, :, 2] / 2).unsqueeze(2)
    o_br_y = (output_bbox[:, :, 1] + output_bbox[:, :, 3] / 2).unsqueeze(2)

    t_tl_x = (target_bbox[:, :, 0] - target_bbox[:, :, 2] / 2).unsqueeze(1)
    t_tl_y = (target_bbox[:, :, 1] - target_bbox[:, :, 3] / 2).unsqueeze(1)
    t_br_x = (target_bbox[:, :, 0] + target_bbox[:, :, 2] / 2).unsqueeze(1)
    t_br_y = (target_bbox[:, :, 1] + target_bbox[:, :, 3] / 2).unsqueeze(1)

    # C (прямоугольник) - пересечение, координаты:
    c_tl_x = torch.max(o_tl_x, t_tl_x)
    c_tl_y = torch.max(o_tl_y, t_tl_y)
    c_br_x = torch.min(o_br_x, t_br_x)
    c_br_y = torch.min(o_br_y, t_br_y)

    # c_br_x > c_tl_x && c_br_y > c_tl_y, area = (c_br_x - c_tl_x) * (c_br_y - c_tl_y)
    # избегаем if, используя умножение на 0
    c_area = torch.max(c_br_x - c_tl_x, torch.zeros_like(c_br_x)) * torch.max(c_br_y - c_tl_y, torch.zeros_like(c_br_y))

    # площадь bbox'ов, h*w
    o_area = (output_bbox[:, :, 2] * output_bbox[:, :, 3]).unsqueeze(2)
    t_area = (target_bbox[:, :, 2] * target_bbox[:, :, 3]).unsqueeze(1)

    # суммарная площадь (пересечение не учтено)
    sum_area = o_area + t_area

    # расчёт отношения "пересечение / объединение"
    iou = c_area / (sum_area - c_area + _eps)  # todo: мб убрать снизу c_area?

    # DIoU метрика (Distance + Intersection over Union)
    score = 1 - iou + distance_diff / (iou + 1)

    # учёт отношения w/h; модифицировавнная CIoU метрика
    aspect_ratio_output = torch.arctan((output_bbox[:, :, 2] + _eps) / (output_bbox[:, :, 3] + _eps)).unsqueeze(2)
    aspect_ratio_target = torch.arctan((target_bbox[:, :, 2] + _eps) / (target_bbox[:, :, 3] + _eps)).unsqueeze(1)
    u_score = (aspect_ratio_output - aspect_ratio_target) ** 2 * 4 / (torch.pi ** 2)
    alpha_u = u_score / (1 - iou + u_score)
    score += alpha_u * u_score

    # расчёт маски не фиктивных target значений через проверку присутствия хотя бы одного значения класса
    real_target_mask = torch.any(target_class, dim=-1).float()  # .bool()
    real_target_mask = real_target_mask.unsqueeze(-2).repeat(1, score.shape[-2], 1)

    # расчёт маски
    with torch.no_grad():
        mask = align(score.detach().clone(), real_target_mask)

    # значения для минимизации
    masked_score = score * mask

    # среднее значение iou для всех target_bbox
    with torch.no_grad():
        masked_iou = iou.detach().clone() * mask
        avg_target_iou = torch.sum(masked_iou) / (torch.sum(real_target_mask.clone() * mask) + _eps)

    # я могу mask умножить на target_class, продублированные для каждого output
    # 0 1 0   [0 1 0] [1 1 0] [0 0 1]    [1 1 0]
    # 1 0 0 * [0 1 0] [1 1 0] [0 0 1] -> [0 1 0]
    # 0 0 0   [0 1 0] [1 1 0] [0 0 1]    [0 0 0]
    # потом сложить их для каждого output. полученный результат и будет ожидаемым значением для каждого output

    # n - кол-во output, m - кол-во target. вроде.
    # ровно то, что нужно, только сложночитаемо. mask [b, n, m], target [b, m, cl]
    with torch.no_grad():
        target_class_masked = torch.einsum('bnm,bmc->bnc', mask, target_class)

    class_loss = torch.nn.functional.binary_cross_entropy_with_logits(output_class, target_class_masked)
    bbox_loss = torch.sum(masked_score)

    # class_loss не включается до тех пор, пока bbox_loss не перейдёт необходимый порог
    if avg_target_iou < threshold_iou_for_class_loss:
        final_loss = bbox_loss
    else:
        final_loss = bbox_loss * k_bbox_loss + class_loss * k_class_loss

    #mAP addon
    class_threshold = 0.7
    with torch.no_grad():
        output_class_prob = torch.nn.functional.sigmoid(output_class)
        _FP = (output_class_prob > class_threshold).float() - target_class_masked.float() #размерности сходятся
        FP = _FP * (_FP > 0).float()
        FN = _FP * (_FP < 0).float() * (-1)
        TP = (output_class_prob > class_threshold).float() * target_class_masked.float()
        #FP, FN, TP = torch.sum(FP), torch.sum(FN), torch.sum(TP)
        FP, FN, TP = torch.sum(FP.reshape(-1, num_classes), dim = 0), torch.sum(FN.reshape(-1, num_classes), dim = 0), torch.sum(TP.reshape(-1, num_classes), dim = 0)
        precision = (TP + _eps) / (TP + FP + _eps)
        recall = (TP + _eps) / (TP + FN + _eps)

    return final_loss, avg_target_iou, class_loss.detach().clone(), precision, recall
