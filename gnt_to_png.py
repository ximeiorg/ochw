import os
from pathlib import Path
import struct
from PIL import Image
def write_txt(save_path: str, content: list, mode='w'):
    """
    将list内容写入txt中
    @param
    content: list格式内容
    save_path: 绝对路径str
    @return:None
    """
    with open(save_path, mode, encoding='utf-8') as f:
        for value in content:
            f.write(value + '\n')



path = 'data/Gnt1.0TrainPart2'
path3 = 'data/Gnt1.0TrainPart3'
save_dir = 'data/train'  # 目录下均为gnt文件

gnt_paths = list(Path(path).iterdir())
gnt_paths.extend(list(Path(path3).iterdir()))

label_list = []
for gnt_path in gnt_paths:
    count = 0
    print(gnt_path)
    with open(str(gnt_path), 'rb') as f:
        while f.read(1) != "":
            f.seek(-1, 1)
            count += 1
            try:
                # 只所以添加try，是因为有时f.read会报错 struct.error: unpack requires a buffer of 4 bytes
                # 原因尚未找到
                length_bytes = struct.unpack('<I', f.read(4))[0]

                tag_code = f.read(2)

                width = struct.unpack('<H', f.read(2))[0]

                height = struct.unpack('<H', f.read(2))[0]

                im = Image.new('RGB', (width, height))
                img_array = im.load()
                for x in range(0, height):
                    for y in range(0, width):
                        pixel = struct.unpack('<B', f.read(1))[0]
                        img_array[y, x] = (pixel, pixel, pixel)

                filename = str(count) + '.png'
                tag_code = tag_code.decode('gbk').strip('\x00')
                save_path = f'{save_dir}/images/{gnt_path.stem}'
                if not Path(save_path).exists():
                    Path(save_path).mkdir(parents=True, exist_ok=True)
                im.save(f'{save_path}/{filename}')

                label_list.append(f'{gnt_path.stem}/{filename}\t{tag_code}')
            except:
                break

write_txt(f'{save_dir}/gt.txt', label_list)