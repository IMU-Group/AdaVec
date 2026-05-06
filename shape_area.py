import torch


def cubic_bezier(P0, P1, P2, P3, t):
    """
    计算三次贝塞尔曲线上的点。

    参数:
    P0, P1, P2, P3 (torch.Tensor): 控制点，每个点是一个形状为 (n,) 的张量，其中 n 是维度数。
    t (torch.Tensor or float): 参数 t，取值范围为 [0, 1]。如果是一个标量，则会被广播到与 P0, P1, P2, P3 兼容的形状。

    返回:
    torch.Tensor: 曲线上的点，形状与 P0, P1, P2, P3 的维度数相同。
    """
    # 确保 t 是一个张量
    # if not isinstance(t, torch.Tensor):
    #     t = torch.tensor(t)

    # 计算 (1-t)^3, 3(1-t)^2*t, 3(1-t)*t^2, t^3
    one_minus_t = 1 - t
    term1 = one_minus_t ** 3
    term2 = 3 * (one_minus_t ** 2) * t
    term3 = 3 * one_minus_t * (t ** 2)
    term4 = t ** 3

    # 计算贝塞尔曲线上的点
    B_t = term1 * P0 + term2 * P1 + term3 * P2 + term4 * P3

    return B_t


def shapes_area(all_shape_points):
    # print(all_shape_points)
    P0 = all_shape_points[..., :-1:3, :]
    P1 = all_shape_points[..., 1::3, :]
    P2 = all_shape_points[..., 2::3, :]
    P3 = all_shape_points[..., 3::3, :]
    # print("P0:", P0.shape)
    # print("P1:", P1.shape)
    # print("P2:", P2.shape)
    # print("P3:", P3.shape)
    P_25 = cubic_bezier(P0, P1, P2, P3, 0.25)
    P_50 = cubic_bezier(P0, P1, P2, P3, 0.5)
    P_75 = cubic_bezier(P0, P1, P2, P3, 0.75)
    # print("P_25:", P_25.shape)
    # print("P_50:", P_50.shape)
    # print("P_75:", P_75.shape)
    vertices = torch.cat([P0, P_25, P_50, P_75, P3], dim=-1).view(all_shape_points.shape[0],-1, 2)
    # print("vertices:", vertices)
    #
    # # 提取x和y坐标
    x = vertices[..., 0]
    # print(x.shape)
    y = vertices[..., 1]
    torch.roll(x, shifts=1,dims=-1)
    #
    # # 使用向量化操作计算面积
    # # 使用torch.roll来循环位移张量中的元素
    area=torch.einsum('ij,ij->i', [x, torch.roll(y, 1)])-torch.einsum('ij,ij->i', [y, torch.roll(x, 1)])

    # 返回面积的绝对值的一半
    return torch.abs(area) / 2


if __name__ == '__main__':
    all_shape_points = torch.randn(5, 12, 2)*100
    area=shapes_area(all_shape_points)
    print(area)

    # 示例使用
    # vertices = [(0, 0), (2, 1), (4, 0), (3, 2), (4, 4), (2, 3), (0, 4), (1, 2), (1, 2)]
    # print("多边形的面积为:", polygon_area(vertices))
