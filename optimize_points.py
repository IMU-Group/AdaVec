from path_simple import path_simple
import torch
import pydiffvg
from tqdm import tqdm
import time

def chamfer_distance(A, B):
    # A: [n, 2] tensor
    # B: [m, 2] tensor

    # 计算A到B的最小距离
    # A.unsqueeze(1) -> [n, 1, 2]
    # B.unsqueeze(0) -> [1, m, 2]
    # broadcast后 -> [n, m, 2]
    # 计算每对点的平方距离 -> [n, m]
    A_B_dist = torch.cdist(A, B, p=2)  # Euclidean distance, [n, m]

    # A中每个点到B中最近的点的距离
    A_to_B = torch.min(A_B_dist, dim=1)[0]  # [n]

    # B中每个点到A中最近的点的距离
    B_to_A = torch.min(A_B_dist, dim=0)[0]  # [m]

    # 倒角距离为两个方向最小距离的和
    chamfer_dist = torch.sum(A_to_B) + torch.sum(B_to_A)

    return chamfer_dist


def render_based_on_shapes_and_shape_groups(shapes, shape_groups, device,
                                            no_grad=True, canvas_width=256,
                                            canvas_height=256, ):
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width=canvas_width, canvas_height=canvas_height, shapes=shapes, shape_groups=shape_groups)
    if no_grad:
        with torch.no_grad():
            render = pydiffvg.RenderFunction.apply
            img = render(canvas_width,  # width
                         canvas_height,  # height
                         2,  # num_samples_x
                         2,  # num_samples_y
                         0,  # seed
                         None,
                         *scene_args)
    else:
        render = pydiffvg.RenderFunction.apply
        img = render(canvas_width,  # width
                     canvas_height,  # height
                     2,  # num_samples_x
                     2,  # num_samples_y
                     0,  # seed
                     None,
                     *scene_args)
    return img.to(device)


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


def polygonal_point(shape_points, sample_num=20):
    # print("shape_points.shape:", shape_points.shape)
    P0 = shape_points[:-1:3, :]
    P1 = shape_points[1::3, :]
    P2 = shape_points[2::3, :]
    P3 = shape_points[3::3, :]
    # print("P0:", P0.shape)
    # print("P1:", P1.shape)
    # print("P2:", P2.shape)
    # print("P3:", P3.shape)
    sample_points = [P0]
    for i in range(1, sample_num):
        sample_points.append(cubic_bezier(P0, P1, P2, P3, i / sample_num))
    sample_points.append(P3)
    vertices = torch.cat(sample_points, dim=-1).view(-1, 2)

    return vertices


def optimize_points(device, canvas_width, canvas_height, shape, shape_save, num_iter=1000):
    points_vars = []
    shape.points.requires_grad = True
    points_vars.append(shape.points)
    # pbar = tqdm(range(num_iter))
    points_optim = torch.optim.Adam(points_vars, lr=1)

    for t in range(num_iter):
        points_optim.zero_grad()
        points_vertices = polygonal_point(torch.cat((shape.points, shape_save.points[:1, :]), dim=0), sample_num=10)[
                          :-1, :]
        save_points_vertices = polygonal_point(torch.cat((shape_save.points, shape_save.points[:1, :]), dim=0),
                                               sample_num=10)[
                               :-1, :]
        loss = chamfer_distance(points_vertices, save_points_vertices)
        loss.backward()
        points_optim.step()
        if t%50==0:
            for param_group in points_optim.param_groups:
                param_group['lr'] = param_group['lr']/1.1
        # pbar.set_postfix({"loss": f"{loss.item()}"})
        # if t%10==0:
        #     shapes = [pydiffvg.Path(num_control_points=torch.tensor(shape.points.shape[0] // 3 * [2]),
        #                             points=shape.points,
        #                             stroke_width=torch.tensor(0.),
        #                             is_closed=True)]
        #     shape_groups = [pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
        #                                         fill_color=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        #                                         stroke_color=None)]
        #     pydiffvg.save_svg(f"result/{t}.svg", canvas_width, canvas_height,
        #                       shapes, shape_groups)
    return shape.points


if __name__ == "__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # canvas_width, canvas_height, shape, shape_save, _ = path_simple("img.png")
    # optimize_shapes(device, canvas_width, canvas_height, shape, shape_save, num_iter=1000)

    from main import get_sam_shapes

    start_time = time.time()
    image_path = "/root/autodl-tmp/img2vec/sam/25.png"
    sam_save_image_path = 'sam_result/25.png'
    sam_save_svg_path_no_point_opt = 'sam_result/25_no_point_opt.svg'
    sam_save_svg_path_point_opt = 'sam_result/25_point_opt.svg'
    save_image_path = 'old_code/result/25.png'
    save_svg_path = 'old_code/result/25.svg'
    path_num = 18
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    w, h, sam_shapes, sam_shape_groups, sam_shapes_save, sam_shape_groups_save = get_sam_shapes(
        image_path=image_path,
        sam_checkpoint=r"/root/autodl-tmp/models/sam_vit_h_4b8939.pth")

    sam_len = len(sam_shapes)

    # sam_scene_args = pydiffvg.RenderFunction.serialize_scene(
    #     canvas_width=w, canvas_height=h, shapes=sam_shapes_save, shape_groups=sam_shape_groups_save)
    # 渲染图像
    # render = pydiffvg.RenderFunction.apply
    # sam_img = render(w, h, 2, 2, 0, None, *sam_scene_args)

    pydiffvg.save_svg(sam_save_svg_path_no_point_opt, w, h,
                      sam_shapes, sam_shape_groups)

    # 保存图像
    # pydiffvg.imwrite(sam_img.cpu(), sam_save_image_path, gamma=1.0)

    time1 = time.time()
    print(f"SAM运行时长: {time1 - start_time} 秒")

    for i in range(sam_len):
        sam_shapes[i].points=optimize_points(device, w, h, sam_shapes[i], sam_shapes_save[i], num_iter=200)
    time2 = time.time()
    print(f"优化控制点时长: {time2 - time1} 秒")
    pydiffvg.save_svg(sam_save_svg_path_point_opt, w, h,
                      sam_shapes, sam_shape_groups)


