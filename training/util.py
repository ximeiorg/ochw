
from PIL import Image


def resize_to_sqr(img: Image):
    # 获取图片的宽高
    width, height = img.size
    # 以最长边为基准，创建白色背景的方形图片
    max_size = max(width, height)
    new_img = Image.new('RGB', (max_size, max_size), (255, 255, 255))
    # 将原图片居中放置在方形背景上
    new_img.paste(img, ((max_size - width) // 2, (max_size - height) // 2))
    # 保存方形图片
    return new_img



if __name__ == '__main__':
    
    img_path = "./data/test/images/020-t/1.png"

    img = Image.open(img_path)
    img = resize_to_sqr(img)
    img.save("test.png")