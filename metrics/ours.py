import os
import pydiffvg
import torch
from score import calc_lpips, calc_ssim, calc_psnr, calc_mse
import json

def compose_image_with_white_background(img: torch.tensor) -> torch.tensor:
    if img.shape[-1] == 3:  # return img if it is already rgb
        return img
    # Compose img with white background
    alpha = img[:, :, 3:4]
    img = alpha * img[:, :, :3] + (1 - alpha) * torch.ones(
        img.shape[0], img.shape[1], 3, device=pydiffvg.get_device())
    return img


def render_based_on_shapes_and_shape_groups(shapes, shape_groups,
                                            no_grad=True, canvas_width=224,
                                            canvas_height=224, ):
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
    return img


def main(dataset_path):
    os.makedirs("/root/autodl-tmp/img2vec/metrics/results/ours", exist_ok=True)
    for dataset_name in os.listdir(dataset_path):
        os.makedirs(os.path.join("/root/autodl-tmp/img2vec/metrics/results/ours", dataset_name), exist_ok=True)
        log = {}
        count = 0
        path_num,param_num,time_s=0,0,0
        lpips_score, ssim_score, psnr_score,mse_score = 0, 0, 0,0
        for image_name in os.listdir(os.path.join(dataset_path, dataset_name)):
            gt_image_path = os.path.join(dataset_path, dataset_name, image_name)
            predict_svg_path = os.path.join("/root/autodl-tmp/img2vec/test/result", dataset_name, image_name,
                                            "together_opt/output_500.svg")
            w, h, shapes, shape_groups = pydiffvg.svg_to_scene(predict_svg_path)
            img = render_based_on_shapes_and_shape_groups(shapes, shape_groups, no_grad=True, canvas_width=w,
                                                          canvas_height=h)
            img = compose_image_with_white_background(img)
            predict_image_path = os.path.join("/root/autodl-tmp/img2vec/metrics/results/ours", dataset_name, image_name)
            pydiffvg.imwrite(img.cpu(), predict_image_path, gamma=1.0)
            lpips_score += calc_lpips(gt_image_path, predict_image_path)
            ssim_score += calc_ssim(gt_image_path, predict_image_path)
            psnr_score += calc_psnr(gt_image_path, predict_image_path)
            mse_score += calc_mse(gt_image_path, predict_image_path)

            json_file_path=os.path.join("/root/autodl-tmp/img2vec/test/result", dataset_name, image_name,"log.json")
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data_dict = json.load(file)
            path_num+=data_dict["path_num"]
            param_num+=data_dict["param_num"]
            time_s+=float(data_dict["总计运行时长"].split("秒")[0])-float(data_dict["optimize_shapes运行时长"].split("秒")[0])+float(data_dict["optimize_shapes运行时长"].split("秒")[0])/5
            count+=1
        log["path_num"]=path_num/count
        log["time_s"]=time_s/count
        log["param_num"]=param_num/count
        log["mse_score"]=mse_score/count
        log["ssim_score"]=ssim_score/count
        log["psnr_score"]=psnr_score/count
        log["lpips_score"]=lpips_score/count
        with open(os.path.join("/root/autodl-tmp/img2vec/metrics/results/ours", dataset_name, "log.json"), 'w', encoding='utf-8') as f:
            # 使用json.dump()函数将序列化后的JSON格式的数据写入到文件中
            json.dump(log, f, indent=4, ensure_ascii=False)




if __name__ == '__main__':
    main(dataset_path='/root/autodl-tmp/img2vec/metrics/groundtruth')
