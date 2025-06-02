import torch

from core.util.draw_output import show_new_img_bbox


# Для каждой матрицы brc[i] удалить строку row_idx и столбец col_idx, указанные в row_indices[i] и col_indices[i] соответственно
def remove_r_c_batched(brc, row_indices, col_indices):
    b, n, m = brc.shape

    batch_indices = torch.arange(b)

    rows_mask = torch.ones(b, n, dtype=torch.bool)
    rows_mask[batch_indices, row_indices] = False
    cols_mask = torch.ones(b, m, dtype=torch.bool)
    cols_mask[batch_indices, col_indices] = False

    # Применяем маски ко всему тензору сразу
    result = brc[rows_mask.unsqueeze(-1).expand(-1, -1, m) & cols_mask.unsqueeze(1).expand(-1, n, -1)]
    result = result.reshape(b, n - 1, m - 1)

    return result


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
def loss_v2(output, target, num_classes=10, distance_k=1):
    # output_bbox = output
    output_bbox, output_class = output.split((4, num_classes), 2)
    target_bbox, target_class = target.split((4, num_classes), 2)

    # todo: часть кода почему-то ломается при отрицательных входных координатах в bbox'ах!
    # можно пофиксить, сдвинув все ббоксы так, чтобы крайнее их минимальное значение было больше 0
    # with torch.no_grad():
    #     min_shift = torch.abs(torch.minimum(torch.min(output_bbox), torch.min(target_bbox))) + 1
    #
    #     shift_output_expanded = torch.zeros_like(output_bbox)
    #     shift_output_expanded[:, :, 0:2] = min_shift.repeat(2)
    #
    #     shift_target_expanded = torch.zeros_like(target_bbox)
    #     shift_target_expanded[:, :, 0:2] = min_shift.repeat(2)
    #
    # output_bbox = output_bbox + shift_output_expanded
    # target_bbox = target_bbox + shift_target_expanded

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
    iou = c_area / (sum_area - c_area)  # todo: мб убрать снизу c_area?

    # DIoU метрика (Distance + Intersection over Union)
    score = 1 - iou + distance_diff / (
            iou + 1)  # * distance_k #(1 - iou) * distance_k + (distance_diff * distance_k) ** 2
    # * 1 / ((1 - iou) + 1)

    # todo: экспериментальная часть: учёт отношения w/h
    _eps = 1e-3  # избежать деления на 0 #считается, что входные данные >=0
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

    # небольшой дебаг
    with torch.no_grad():
        iou_score = iou.detach().clone() * mask
        avg_target_iou = torch.sum(iou_score) / torch.sum(real_target_mask.clone() * mask)

    # todo: учёт классов, разные лоссы к полученным значениям, гиперпараметры

    # я могу score умножить на target_class, продублированные для каждого output
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

    #print(class_loss)
    #print(bbox_loss)

    return bbox_loss + class_loss, avg_target_iou, class_loss.detach().clone()


def simple_search(num_iter=300):
    example_scale = 4
    image = torch.ones(3, 224 * example_scale, 224 * example_scale) * 255

    # outputb = torch.nn.Parameter(torch.tensor([[
    #     [0.1, 0.1, 0.1, 0.1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #
    #     [0.1, 0.1, 0.1, 0.1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #
    #     [0.1, 0.1, 0.1, 0.1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # ]]))
    #
    # targetb = torch.tensor([[
    #     [0.5, 0.5, 0.4, 0.2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #
    #     [0.7, 0.1, 0.2, 0.2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #
    #     [0.3, 0.7, 0.1, 0.3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # ]])

    outputb = torch.nn.Parameter(
        torch.rand(2, 10, 10 + 4))  # с randn что-то ломается.. (скорее всего из-за... бля, отрицательные w,h bbox'ов..)
    targetb = torch.cat((torch.rand(2, 10, 4),
                         torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).unsqueeze(0).unsqueeze(0).repeat(2, 10, 1)),
                        dim=-1)
    targetb[:, :, 0:2] = torch.rand(2, 10, 2) * 2 - 1
    targetb[0, -2:, 4:] = 0

    optimizer = torch.optim.Adam([outputb], lr=0.006)

    def draw(scale=2, debug_negative=True, save_path=None):
        with torch.no_grad():
            showed_targetb = targetb.clone()
            showed_targetb[:, :, 0:2] = (showed_targetb[:, :, 0:2] + scale / (1 if debug_negative else 2)) / (scale * 2)
            showed_targetb[:, :, 2:4] = showed_targetb[:, :, 2:4] / (scale * 2)
            showed_outputb = outputb.clone()
            showed_outputb[:, :, 0:2] = (showed_outputb[:, :, 0:2] + scale / (1 if debug_negative else 2)) / (scale * 2)
            showed_outputb[:, :, 2:4] = showed_outputb[:, :, 2:4] / (scale * 2)

            for i in range(showed_targetb.shape[0]):
                show_new_img_bbox(image, showed_targetb[i], showed_outputb[i], save_path=save_path)
                if save_path:  # сохраняем ток первую
                    break

    # draw()

    for i in range(num_iter):
        loss, avg_iou_err, _ = loss_v2(outputb, targetb)  # check iou. strange results
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(avg_iou_err.item())

        draw(save_path=f"/home/dima/Projects/ssm-recognition/data/experiments/loss/img_{i:06d}.png")

    # draw()


def simple_example():
    example_scale = 1
    image = torch.ones(3, 224 * example_scale, 224 * example_scale) * 255

    outputb = torch.tensor([[
        [0.1, 0.1, 0.1, 0.1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0.8, 0.2, 0.1, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0.3, 0.7, 0.2, 0.1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    ]])
    targetb = torch.tensor([[
        [0.2, 0.2, 0.2, 0.2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0.4, 0.7, 0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]])

    outputb.requires_grad = True

    loss, iou_score, _ = loss_v2(outputb, targetb)

    # print(iou_score)

    show_new_img_bbox(image, targetb.squeeze(0), outputb.squeeze(0))


if __name__ == "__main__":
    simple_example()
    # simple_search()
    # exit(0)
