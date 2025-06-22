import json

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import vtracer
import pydiffvg
import torch
from skimage import io
from skimage.segmentation import felzenszwalb
from skimage.measure import label
import concurrent.futures
import asyncio
from tqdm import tqdm
import skimage
import skimage.io
from geometric_loss import GeometryLoss
from path_simple import path_simple
import copy
import random
from shape_area import shapes_area
from optimize_points import optimize_points
from scipy.ndimage import binary_fill_holes
from skimage import io, segmentation, color
from sklearn.cluster import DBSCAN
import os
from scipy.ndimage import binary_fill_holes
import cv2
import time


def get_sdf(phi, method='skfmm', **kwargs):
    if method == 'skfmm':
        import skfmm
        phi = (phi - 0.5) * 2
        if (phi.max() <= 0) or (phi.min() >= 0):
            return np.zeros(phi.shape).astype(np.float32)
        sd = skfmm.distance(phi, dx=1)

        flip_negative = kwargs.get('flip_negative', True)
        if flip_negative:
            sd = np.abs(sd)

        truncate = kwargs.get('truncate', 10)
        sd = np.clip(sd, -truncate, truncate)
        # print(f"max sd value is: {sd.max()}")

        zero2max = kwargs.get('zero2max', True)
        if zero2max and flip_negative:
            sd = sd.max() - sd
        elif zero2max:
            raise ValueError

        normalize = kwargs.get('normalize', 'sum')
        if normalize == 'sum':
            sd /= sd.sum()
        elif normalize == 'to1':
            sd /= sd.max()
        return sd


def get_bezier_circle(radius=1, segments=4, bias=None):
    points = []
    if bias is None:
        bias = (random.random(), random.random())
    avg_degree = 360 / (segments * 3)
    for i in range(0, segments * 3):
        point = (np.cos(np.deg2rad(i * avg_degree)),
                 np.sin(np.deg2rad(i * avg_degree)))
        points.append(point)
    points = torch.tensor(points)
    points = (points) * radius + torch.tensor(bias).unsqueeze(dim=0)
    points = points.type(torch.FloatTensor)
    return points


def compute_sine_theta(s1, s2):  # s1 and s2 aret two segments to be uswed
    # s1, s2 (2, 2)
    v1 = s1[1, :] - s1[0, :]
    v2 = s2[1, :] - s2[0, :]
    # print(v1, v2)
    sine_theta = (v1[0] * v2[1] - v1[1] * v2[0]) / (torch.norm(v1) * torch.norm(v2))
    return sine_theta


def Xing_Loss(shapes, scale=1):  # x[ npoints,2]
    loss = 0.
    # print(len(x_list))
    for shape in shapes:
        x = shape.points
        seg_loss = 0.
        N = x.size()[0]
        x = torch.cat([x, x[0, :].unsqueeze(0)], dim=0)  # (N+1,2)
        segments = torch.cat([x[:-1, :].unsqueeze(1), x[1:, :].unsqueeze(1)], dim=1)  # (N, start/end, 2)
        assert N % 3 == 0, 'The segment number is not correct!'
        segment_num = int(N / 3)
        for i in range(segment_num):
            cs1 = segments[i * 3, :, :]  # start control segs
            cs2 = segments[i * 3 + 1, :, :]  # middle control segs
            cs3 = segments[i * 3 + 2, :, :]  # end control segs
            # print('the direction of the vectors:')
            # print(compute_sine_theta(cs1, cs2))
            direct = (compute_sine_theta(cs1, cs2) >= 0).float()
            opst = 1 - direct  # another direction
            sina = compute_sine_theta(cs1, cs3)  # the angle between cs1 and cs3
            seg_loss += direct * torch.relu(- sina) + opst * torch.relu(sina)
            # print(direct, opst, sina)
        seg_loss /= segment_num

        templ = seg_loss
        loss += templ * scale  # area_loss * scale

    return loss / (len(shapes))


def get_sam_seg(image_path, sam_checkpoint, exp_path):
    os.makedirs(os.path.join(exp_path, "sam"), exist_ok=True)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    model_type = "default"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    if len(masks) == 0:
        return [], []
    sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print("sam分割数量:", len(sorted_anns))
    sam = []
    for c, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        # masked_image = np.zeros_like(image)
        masked_image = mask * 255
        fill_color = image[mask == 1].mean(axis=0).astype(np.uint8)
        hole_filled_image = binary_fill_holes(mask * 255).astype(np.uint8) * 255
        # print(hole_filled_image.dtype)
        num_labels, labels = cv2.connectedComponents(hole_filled_image)
        for label in range(1, num_labels):  # 从1开始，因为0表示背景
            # colored_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_image = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            mask_image[labels == label] = 1
            # colored_image[labels == label] = fill_color
            area = np.sum(mask_image)
            sam.append({"area": area, "fill_color": fill_color, "mask_image": mask_image})
    sam = sorted(sam, key=lambda x: x['area'], reverse=True)
    for i in range(len(sam)):
        # print("fill_color:",type(sam[i]["fill_color"]))
        # print("mask_image:",sam[i]["mask_image"].shape)
        masked_image_temp = np.zeros_like(image)
        # print("masked_image:",masked_image_temp.shape)
        masked_image_temp[sam[i]["mask_image"]==1] = sam[i]["fill_color"]
        masked_image_temp = cv2.cvtColor(masked_image_temp, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(exp_path, "sam", f"{i}.png"), masked_image_temp)
        # cv2.imwrite(os.path.join(exp_path, "sam", f"{i}.png"), sam[i]["mask_image"] * 255)
    return sam


def get_clus_seg(image_path, exp_path):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[2] == 4:
        # 如果有 Alpha 通道，提取 RGB 通道
        image = image[:, :, :3]
    else:
        # 如果没有 Alpha 通道，直接使用原图
        image = image
    os.makedirs(os.path.join(exp_path, "clus"), exist_ok=True)
    # 使用SLIC进行超像素分割
    n_segments = 1000  # 超像素数量
    compactness = 10  # 紧凑性参数
    segments = segmentation.slic(image, n_segments=n_segments, compactness=compactness, start_label=1)

    # io.imsave(f'result/{file_name}/slic.png',(segmentation.mark_boundaries(image, segments)*255).astype(np.uint8))
    # 计算每个超像素的RGB均值
    labels = np.unique(segments)
    # print(len(labels))
    avg_colors = np.array([image[segments == label].mean(axis=0) for label in labels])
    # print(len(avg_colors))

    # 使用DBSCAN进行聚类
    # 图形参数
    db = DBSCAN(eps=10, min_samples=1).fit(avg_colors)
    # db = DBSCAN(eps=4.5, min_samples=1).fit(avg_colors)
    cluster_labels = db.labels_
    # print(np.unique(cluster_labels))

    # 根据聚类结果生成分割图像
    mask_images = [np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8) for _ in
                   range(len(np.unique(cluster_labels)))]
    for i, label in enumerate(labels):
        if cluster_labels[i] != -1:  # -1 表示噪声点
            mask_images[cluster_labels[i]][segments == label] = 1

    clus = []
    for i, mask in enumerate(mask_images):
        fill_color = image[mask == 1].mean(axis=0).astype(np.uint8)
        hole_filled_image = binary_fill_holes(mask * 255).astype(np.uint8) * 255
        num_labels, labels = cv2.connectedComponents(hole_filled_image)
        for label in range(1, num_labels):  # 从1开始，因为0表示背景
            # colored_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_image = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
            mask_image[labels == label] = 1
            # colored_image[labels == label] = fill_color
            area = np.sum(mask_image)
            clus.append({"area": area, "mask_image": mask_image, "fill_color": fill_color})
        clus = sorted(clus, key=lambda x: x['area'], reverse=True)
    for i in range(len(clus)):
        masked_image = np.zeros_like(image)
        masked_image[clus[i]["mask_image"]==1] = clus[i]["fill_color"]
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(exp_path, "clus", f"{i}.png"), masked_image)
        # cv2.imwrite(os.path.join(exp_path, "clus", f"{i}.png"), clus[i]["mask_image"] * 255)
    return clus


def intersection_over_union(img1, img2):
    # 确保输入图像为二值化图像
    assert img1.shape == img2.shape, "Images must have the same dimensions"

    # 计算交集
    intersection = np.sum(np.logical_and(img1, img2))
    # 计算并集
    union = np.sum(np.logical_or(img1, img2))

    # 计算交并比
    iou = intersection / union if union > 0 else 0  # 避免除以零的情况

    return iou


def merge(sam, clus, exp_path):
    os.makedirs(os.path.join(exp_path, "merge"), exist_ok=True)
    img_list = []
    for img_dict in clus:
        if img_dict["area"] > 40:
            img_list.append(
                {"mask_image": img_dict["mask_image"], "area": img_dict["area"], "fill_color": img_dict["fill_color"],
                 "source": "clus"})
    for img_dict in sam:
        if img_dict["area"] > 10:
            img_list.append(
                {"mask_image": img_dict["mask_image"], "area": img_dict["area"], "fill_color": img_dict["fill_color"],
                 "source": "sam"})
    img_list = sorted(img_list, key=lambda x: x["area"], reverse=True)
    mask_image_tag = np.zeros((img_list[0]["mask_image"].shape[0], img_list[0]["mask_image"].shape[1])).astype(int) - 1
    merge_res = []
    # def iou(i,j):
    #     Intersection = np.sum(img_list[i]["img"][img_list[j]["img"]])
    #     union = np.sum(img_list[i]["img"]) + np.sum(img_list[j]["img"]) - Intersection
    #     print(Intersection,union)
    #     return Intersection/union
    for i in range(len(img_list)):
        # print(img_list[i])
        if img_list[i]["source"] == "clus":
            extent = list(range(max(0, i - 2), min(len(img_list) - 1, i + 2) + 1))
            extent.remove(i)
            IOU = [intersection_over_union(img_list[i]["mask_image"], img_list[j]["mask_image"]) for j in extent if
                   img_list[j]["source"] == "sam"]
            # print("iou:", IOU)
            if IOU and max(IOU) > 0.85:
                continue
        mask_image_tag[img_list[i]["mask_image"] == 1] = i
    # print(np.unique(mask_image))
    for i in range(len(img_list)):
        if np.sum(mask_image_tag == i) / np.sum(img_list[i]["mask_image"]) > 0.1:
            merge_res.append(img_list[i])
    merge_res[0]["mask_image"][:]=1
    for i in range(len(merge_res)):
        if merge_res[i]["source"] == "clus":
            cv2.imwrite(os.path.join(exp_path, "merge", f"{i}_clus.png"), merge_res[i]["mask_image"] * 255)
        else:
            cv2.imwrite(os.path.join(exp_path, "merge", f"{i}_sam.png"), merge_res[i]["mask_image"] * 255)
    return merge_res


def get_shapes(merge_res, exp_path):
    shapes = []
    shape_groups = []
    w, h = 255, 255
    for i in range(len(merge_res)):
        mask_image, fill_color = merge_res[i]["mask_image"], merge_res[i]["fill_color"] / 255
        # print("fill_color:",type(fill_color))
        cv2.imwrite(os.path.join(exp_path, f"temp/{i}.png"), mask_image * 255)
        vtracer.convert_image_to_svg_py(os.path.join(exp_path, f"temp/{i}.png"),
                                        os.path.join(exp_path, f"temp/{i}.svg"),
                                        colormode='color',  # ["color"] or "binary"
                                        hierarchical='stacked',  # ["stacked"] or "cutout"
                                        mode='spline',  # ["spline"] "polygon", or "none"
                                        filter_speckle=4,  # default: 4
                                        color_precision=6,  # default: 6
                                        layer_difference=16,  # default: 16
                                        corner_threshold=60,  # default: 60
                                        length_threshold=4.0,  # in [3.5, 10] default: 4.0
                                        max_iterations=10,  # default: 10
                                        splice_threshold=45,  # default: 45
                                        path_precision=3  # default: 8
                                        )
        w, h, shapes_temp, shape_groups_temp = pydiffvg.svg_to_scene(os.path.join(exp_path, f"temp/{i}.svg"))

        if len(shapes_temp) == 1 and i != 0:
            continue

        k = -1
        maxlen = -1
        if i==0:
            for j, shape_temp in enumerate(shapes_temp):
                pathObj_pad = shape_temp.points[:1, :].detach()
                if maxlen <= shapes_area(torch.cat((shape_temp.points, pathObj_pad), dim=0).unsqueeze(0)):
                    k = j
        else:
            for j, shape_temp in enumerate(shapes_temp):
                pathObj_pad = shape_temp.points[:1, :].detach()
                if maxlen >= shapes_area(torch.cat((shape_temp.points, pathObj_pad), dim=0).unsqueeze(0)):
                    k = j

        shapes.append(
            pydiffvg.Path(num_control_points=shapes_temp[k].num_control_points, points=shapes_temp[k].points,
                          stroke_width=torch.tensor(0.),
                          is_closed=shapes_temp[k].is_closed))

        init_opacity = 1.0
        reference_color = torch.tensor(list(fill_color) + [init_opacity], dtype=torch.float32)

        # print("reference_color:",reference_color)

        def bbox2(img):
            rows = np.any(img, axis=1)
            cols = np.any(img, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            return rmin, rmax, cmin, cmax

        rmin, rmax, cmin, cmax = bbox2(mask_image)
        gradient_radius_ref = np.clip((rmax - rmin) / h / 2, 0.1, 0.5), np.clip((cmax - cmin) / w / 2, 0.1, 0.5)
        gradient_r = np.sqrt(gradient_radius_ref[0] * gradient_radius_ref[1])
        center = (np.argwhere(mask_image)).mean(axis=0)
        wref, href = copy.deepcopy(center)
        wref = max(0, min(int(wref), w - 1))
        href = max(0, min(int(href), h - 1))
        canvas_size = (h, w)
        canvas_size_np = np.array(canvas_size, dtype=np.float32)
        canvas_size = torch.FloatTensor(canvas_size_np).requires_grad_(False)
        gradient_params = {
            'center': torch.FloatTensor(np.array([wref, href]) / canvas_size_np) * canvas_size,
            'radius': torch.FloatTensor([gradient_r, gradient_r]) * canvas_size,
            'offsets': torch.FloatTensor([0.0, 1.0]),
            'stop_colors': torch.stack([reference_color, reference_color]),
        }
        fill_color_init = pydiffvg.RadialGradient(
            center=gradient_params['center'],
            radius=gradient_params['radius'],
            offsets=gradient_params['offsets'],
            stop_colors=gradient_params['stop_colors'],
        )
        fill_color_params = list(gradient_params.values())
        path_group = pydiffvg.ShapeGroup(
            shape_ids=torch.LongTensor([len(shapes) - 1]),
            fill_color=fill_color_init,
            stroke_color=None,
        )
        path_group.fill_color_params = fill_color_params
        shape_groups.append(path_group)

    return w, h, shapes, shape_groups


def line_intersection(A, B, C, D):
    # 获取点的坐标
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D
    # print(x1, y1, x2, y2, x3, y3)

    # 构建系数矩阵和常数向量
    A_matrix = torch.tensor([[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]])
    b_vector = torch.tensor([x3 - x1, y3 - y1])

    # 判断是否有唯一解（行列式不为0）
    det = torch.det(A_matrix)
    if det == 0:
        return None  # 平行或共线，没有交点

    # 求解线性方程组，得到参数 t 和 u
    t_u = torch.linalg.solve(A_matrix, b_vector)

    # 计算交点坐标
    t = t_u[0]
    intersection = A + t * (B - A)

    return intersection


def are_on_same_side(A, B, C, D, E, F, G):
    # 将2D点扩展到3D
    A_3d = torch.tensor([A[0], A[1], 0.0])
    G_3d = torch.tensor([G[0], G[1], 0.0])

    # 计算向量AG
    AG = G_3d - A_3d

    # 点列表
    points = [B, C, D, E, F]
    results = []

    for point in points:
        point_3d = torch.tensor([point[0], point[1], 0.0])
        cross_product = torch.linalg.cross(AG, point_3d - A_3d)
        results.append(cross_product[2])

    # 检查所有点的叉积符号是否相同
    signs = [torch.sign(res).item() for res in results]
    return all(s == signs[0] for s in signs)


def angle_between(v1, v2):
    # 计算两个向量的夹角
    cos_theta = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2))
    angle = torch.acos(torch.clamp(cos_theta, -1.0, 1.0)) * 180 / torch.pi  # 限制范围
    return angle


def is_merge(A, B, C, D, E, F, G):
    if not are_on_same_side(A, B, C, D, E, F, G):
        return False
    # 计算各个角的弧度
    angle_DAG = angle_between(D - A, G - A)
    angle_DGA = angle_between(D - G, A - G)
    angle_BAG = angle_between(B - A, G - A)
    angle_FGA = angle_between(F - G, A - G)
    angle_CDE = angle_between(C - D, E - D)
    # print("angle_DAG, angle_DGA, angle_BAG, angle_FGA, angle_CDE:")
    # print(angle_DAG, angle_DGA, angle_BAG, angle_FGA, angle_CDE)
    if angle_DAG < 90 - 10 and angle_DGA < 90 - 10 and angle_DAG +10 < angle_BAG < 90 - 10 and angle_DGA + 10 < angle_FGA < 90 - 10 and 180 - angle_CDE < 8:
        return True
    return False


def shapes_simple(shapes, shape_groups):
    simpled_shapes = []
    simpled_shape_groups = []
    for i in range(len(shapes)):
        points = shapes[i].points
        points = torch.cat([points, points[:1, :]])
        index = 0
        while index + 6 < points.shape[0]:
            if points.shape[0] <= 12:
                break
            A, B, C, D, E, F, G = points[index], points[index + 1], points[index + 2], points[index + 3], points[
                index + 4], \
                points[index + 5], points[index + 6]
            # print(is_merge(A, B, C, D, E, F, G))
            if is_merge(A, B, C, D, E, F, G):
                intersection = line_intersection(A, B, G, F)
                points[index + 1, :] = B + 0 * (intersection - B)
                points[index + 5, :] = F + 0 * (intersection - F)
                points = torch.cat((points[:index + 2], points[index + 5:]), dim=0)
            else:
                index = index + 3
        index = 0
        while index + 3 <= points.shape[0]:
            if points.shape[0] <= 12:
                break
            if torch.norm(points[index] - points[index + 3], p=2) <= 10:
                if index + 4 < points.shape[0]:
                    points = torch.cat((points[:index + 1], points[index + 4:]), dim=0)
                else:
                    points = torch.cat((points[:index], points[:1]), dim=0)
            else:
                index = index + 3
        points = points[:-1, :].detach().clone()
        simpled_shapes.append(pydiffvg.Path(num_control_points=torch.tensor(points.shape[0] // 3 * [2]), points=points,
                                            stroke_width=torch.tensor(0.),
                                            is_closed=shapes[i].is_closed))
        simpled_shape_groups.append(shape_groups[i])
    return simpled_shapes, simpled_shape_groups


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


def compose_image_with_black_background(img: torch.tensor) -> torch.tensor:
    if img.shape[-1] == 3:  # return img if it is already rgb
        return img
    # Compose img with white background
    alpha = img[:, :, 3:4]
    img = alpha * img[:, :, :3] + (1 - alpha) * torch.zeros(
        img.shape[0], img.shape[1], 3, device=pydiffvg.get_device())
    return img


def read_png_image_from_path(path_to_png_image: str) -> torch.tensor:
    numpy_image = skimage.io.imread(path_to_png_image)
    normalized_tensor_image = torch.from_numpy(numpy_image).to(
        torch.float32) / 255.0
    return normalized_tensor_image


def optimize_shapes(target_image_path, shapes, shape_groups, num_iter: int, canvas_width, canvas_height, device,
                    exp_path):
    render = pydiffvg.RenderFunction.apply
    target_image = read_png_image_from_path(target_image_path).to(device)
    big_points_vars = []
    small_points_vars = []
    color_vars = []
    geo_loss = GeometryLoss()
    for i, shape in enumerate(shapes):
        max_len = shape.points.shape[0] + 1
        pathObj_points = shape.points.view(-1, 2)
        pathObj_pad = shape.points[:1, :].detach().repeat(max_len - shape.points.shape[0], 1)
        shape_points = torch.cat((pathObj_points, pathObj_pad), dim=0).unsqueeze(0)
        # print(shape_points)
        area = shapes_area(shape_points).sum()
        # print("area:", area)
        shape.points.requires_grad = True
        big_points_vars.append(shape.points)
        # if area > 5000:
        #     big_points_vars.append(shape.points)
        # else:
        #     small_points_vars.append(shape.points)

    for group in shape_groups:
        for param in group.fill_color_params:
            param.requires_grad_(True)
        color_vars += group.fill_color_params
    big_points_optim = torch.optim.Adam(big_points_vars, lr=0.1)
    # small_points_optim = torch.optim.Adam(small_points_vars, lr=1)
    color_optim = torch.optim.Adam(color_vars, lr=1e-2)
    rendered_image = torch.zeros_like(target_image)
    pbar = tqdm(range(num_iter+1))
    for t in pbar:
        big_points_optim.zero_grad()
        # small_points_optim.zero_grad()
        color_optim.zero_grad()
        rendered_image = render_based_on_shapes_and_shape_groups(
            shapes, shape_groups, device, no_grad=False,
            canvas_width=canvas_width,
            canvas_height=canvas_height)
        if rendered_image.shape[-1] == 4:
            rendered_image = compose_image_with_black_background(
                rendered_image)
        if target_image.shape[-1] == 4:
            target_image = compose_image_with_black_background(target_image)

        # loss = torch.nn.L1Loss()(rendered_image, target_image) + geo_loss.compute(shapes)
        loss = ((rendered_image - target_image) ** 2)
        loss_l1 = loss.sum(2).mean()
        # print("loss_l1:",loss_l1)
        # print(loss.shape)
        # loss = torch.nn.L1Loss()(rendered_image, target_image)
        use_distance_weighted_loss = True
        if use_distance_weighted_loss:
            shapes_forsdf = copy.deepcopy(shapes)
            shape_groups_forsdf = copy.deepcopy(shape_groups)
            for si in shapes_forsdf:
                si.stroke_width = torch.FloatTensor([0]).to(device)
            for sg_idx, sgi in enumerate(shape_groups_forsdf):
                sgi.fill_color = torch.FloatTensor([1, 1, 1, 1]).to(device)
                sgi.shape_ids = torch.LongTensor([sg_idx]).to(device)

            sargs_forsdf = pydiffvg.RenderFunction.serialize_scene(
                w, h, shapes_forsdf, shape_groups_forsdf)
            with torch.no_grad():
                im_forsdf = render(w, h, 2, 2, 0, None, *sargs_forsdf)
            # use alpha channel is a trick to get 0-1 image
            im_forsdf = (im_forsdf[:, :, 3]).detach().cpu().numpy()
            distance_weight = get_sdf(im_forsdf, normalize='to1')
            loss_weight = np.clip(distance_weight, 0.0, 1.0)
            loss_weight = torch.FloatTensor(loss_weight).to(device)
            distance_weighted_loss = (loss.sum(2) * loss_weight).mean()
            # print("distance_weighted_loss:", distance_weighted_loss)
            loss = loss_l1 + distance_weighted_loss
        use_area_loss = False
        if use_area_loss:
            max_len = max([shape.points.shape[0] for shape in shapes]) + 1
            curves = []
            for shape in shapes:
                if shape.points.shape[0] < max_len:
                    pathObj_points = shape.points.view(-1, 2)
                    pathObj_pad = shape.points[:1, :].detach().repeat(max_len - shape.points.shape[0], 1)
                    curves.append(torch.cat((pathObj_points, pathObj_pad), dim=0))
            all_shape_points = torch.stack(curves)
            average_area = shapes_area(all_shape_points).sum() / len(shapes) / canvas_width / canvas_width
            loss = loss + 0.05 * average_area
            # print("average_area:", average_area)
        use_geo_loss = True
        if use_geo_loss:
            g_loss = geo_loss.compute(shapes)
            loss = loss + 0.1 * g_loss

        # print("loss:",loss)
        loss.backward()

        # Take a gradient descent step:
        big_points_optim.step()
        # small_points_optim.step()
        color_optim.step()
        # clamp colors to [0, 1]:
        with torch.no_grad():
            for group in shape_groups:
                if isinstance(group.fill_color, torch.Tensor):
                    group.fill_color.clamp_(0.0, 1.0)
                else:
                    group.fill_color.offsets.clamp_(0.0, 1.0)
                    group.fill_color.stop_colors.clamp_(0.0, 1.0)
        if use_distance_weighted_loss:
            pbar.set_postfix(
                {"loss": f"{loss.item()}", "loss_l1": f"{loss_l1.item()}",
                 "distance_weighted_loss": f"{distance_weighted_loss.item()}"})
        else:
            pbar.set_postfix(
                {"loss": f"{loss_l1.item()}"})
        if t % 50 == 0:
            pydiffvg.save_svg(os.path.join(exp_path, f"together_opt/output_{t}.svg"), canvas_width, canvas_height,
                              shapes, shape_groups)


if __name__ == "__main__":
    image_name="emoji_u1f940.png"
    log={}
    exp_path = os.path.join(os.path.join("/root/autodl-tmp/img2vec/results", image_name.split(".")[0]))
    os.makedirs(exp_path, exist_ok=True)
    image_path = os.path.join("/root/autodl-tmp/img2vec/examples",image_name)
    print("image_path:", image_path)
    # file_name = "51"
    start_time = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # image_path = f"/root/autodl-tmp/img2vec/img/{file_name}.png"
    sam = get_sam_seg(image_path=image_path, sam_checkpoint=r"/root/autodl-tmp/models/sam_vit_h_4b8939.pth",
                      exp_path=exp_path)
    time1 = time.time()
    log["get_sam_seg运行时长"] = f"{time1 - start_time} 秒"
    # print(f"get_sam_seg运行时长: {time1 - start_time} 秒")
    clus = get_clus_seg(image_path=image_path, exp_path=exp_path)
    time2 = time.time()
    log["get_clus_seg运行时长"] = f"{time2 - time1} 秒"
    # print(f"get_clus_seg运行时长: {time2 - time1} 秒")
    merge_res = merge(sam, clus, exp_path=exp_path)
    time3 = time.time()
    log["merge运行时长"] = f"{time3 - time2} 秒"
    # print(f"merge运行时长: {time3 - time2} 秒")
    os.makedirs(os.path.join(exp_path, "temp"), exist_ok=True)
    w, h, shapes, shape_groups = get_shapes(merge_res, exp_path=exp_path)
    time4 = time.time()
    log["get_shapes运行时长"] = f"{time4 - time3} 秒"
    # print(f"get_shapes运行时长: {time4 - time3} 秒")
    pydiffvg.save_svg(os.path.join(exp_path, "save_svg_simple_pre.svg"), w, h, shapes, shape_groups)
    simpled_shapes, simpled_shape_groups = shapes_simple(shapes, shape_groups)
    # simpled_shapes, simpled_shape_groups = shapes, shape_groups
    time5 = time.time()
    log["shapes_simple运行时长"] = f"{time5 - time4} 秒"
    # print(f"shapes_simple运行时长: {time5 - time4} 秒")

    pydiffvg.save_svg(os.path.join(exp_path, "save_simpled_svg.svg"), w, h, simpled_shapes,
                      simpled_shape_groups)
    os.makedirs(os.path.join(exp_path, "shape_opted"), exist_ok=True)
    pbar = tqdm(range(len(shapes)))
    for i in pbar:
        simpled_shapes[i].points = optimize_points(device, w, h, simpled_shapes[i], shapes[i], num_iter=200)
    pydiffvg.save_svg(os.path.join(exp_path, "shape_opted/path_opt.svg"), w, h, simpled_shapes,
                      simpled_shape_groups)

    time6 = time.time()
    log["optimize_points运行时长"] = f"{time6 - time5} 秒"
    # print(f"optimize_points运行时长: {time6 - time5} 秒")

    os.makedirs(os.path.join(exp_path, "together_opt"), exist_ok=True)

    optimize_shapes(target_image_path=image_path, shapes=simpled_shapes, shape_groups=simpled_shape_groups,
                    num_iter=100,
                    canvas_width=w, canvas_height=h, device=device, exp_path=exp_path)

    time7 = time.time()
    log["optimize_shapes运行时长"] = f"{time7 - time6} 秒"
    # print(f"optimize_shapes运行时长: {time7 - time6} 秒")

    end_time = time.time()
    log["总计运行时长"] = f"{end_time - start_time} 秒"
    # print(f"总计运行时长: {end_time - start_time} 秒")
    log["path_num"] = len(shapes)
    control_point_num = 0
    for simpled_shape in simpled_shapes:
        control_point_num += simpled_shape.points.shape[0]
    log["control_point_num"] = control_point_num
    log["param_num"] = control_point_num * 2 + len(shapes) * 12
    with open(os.path.join(exp_path, "log.json"), 'w', encoding='utf-8') as f:
        # 使用json.dump()函数将序列化后的JSON格式的数据写入到文件中
        json.dump(log, f, indent=4, ensure_ascii=False)



