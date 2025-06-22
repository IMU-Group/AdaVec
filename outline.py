import pydiffvg
import torch
shapes = []
shape_groups = []
w, h, shapes_temp, shape_groups_temp = pydiffvg.svg_to_scene(fr"vtrace_outline/12.svg")
for i in range(len(shapes_temp)):
    shapes.append(
        pydiffvg.Path(num_control_points=shapes_temp[i].num_control_points, points=shapes_temp[i].points,
                      stroke_width=torch.tensor(1),
                      is_closed=shapes_temp[i].is_closed,
                      ))
    shape_groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([i]),fill_color=None,
                                   stroke_color=torch.cat([torch.rand(3),torch.tensor([1.0])])))

pydiffvg.save_svg("outline.svg", w, h,shapes, shape_groups)


