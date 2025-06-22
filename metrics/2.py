# coding:utf-8
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import lpips
from skimage.metrics import mean_squared_error
import cv2

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像大小（可选）
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
    ])
    image_tensor = transform(image).unsqueeze(0)  # 增加一个batch维度
    return image_tensor


def calc_ssim(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    ssim_score : numpy.float64
        结构相似性指数（structural similarity index，SSIM）.

    References
    -------
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

    '''
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score


def calc_lpips(img1_path, img2_path):
    # 加载图像
    loss_fn = lpips.LPIPS(net='alex')  # 使用AlexNet作为基础网络，你也可以选择'vgg'
    image1 = load_image(img1_path)
    image2 = load_image(img2_path)
    lpips_score = loss_fn(image1, image2)
    return lpips_score.item()


def calc_psnr(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    psnr_score : numpy.float64
        峰值信噪比(Peak Signal to Noise Ratio, PSNR).

    References
    -------
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    '''
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score

def calc_mse_1(img1_path, img2_path):
    image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # 灰度读取图像
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    image1_resized = cv2.resize(image1, (256, 256))/255
    image2_resized = cv2.resize(image2, (256, 256))/255
    mse = mean_squared_error(image1_resized, image2_resized)
    return mse

def calc_mse_2(img1_path, img2_path):
    image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)  # 灰度读取图像
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    image1_resized = cv2.resize(image1, (256, 256))/255
    image2_resized = cv2.resize(image2, (256, 256))/255
    mse=np.mean((image1_resized - image2_resized) ** 2)
    return mse

if __name__ == '__main__':
    mse_1 = calc_mse_1(img1_path="./results/1.png", img2_path="./results/2.png")
    print(mse_1)
    mse_2 = calc_mse_2(img1_path="./results/1.png", img2_path="./results/2.png")
    print(mse_2)
