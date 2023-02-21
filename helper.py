from cv2 import cv2
import tensorflow as tf
import os
from PIL import Image


def check_images(s_dir, ext_list):
    '''
    检查文件格式是否正确
    '''
    bad_images = []
    bad_ext = []
    s_list = os.listdir(s_dir)
    for klass in s_list:
        klass_path = os.path.join(s_dir, klass)
        print('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list = os.listdir(klass_path)
            for f in file_list:
                f_path = os.path.join(klass_path, f)
                index = f.rfind('.')
                ext = f[index + 1:].lower()
                if ext not in ext_list:
                    print('file ', f_path, ' has an invalid extension ', ext)
                    bad_ext.append(f_path)
                if os.path.isfile(f_path):
                    try:
                        img = cv2.imread(f_path)
                        shape = img.shape
                        image_contents = tf.io.read_file(f_path)
                        image = tf.image.decode_jpeg(image_contents, channels=3)
                    except Exception as e:
                        print('file ', f_path, ' is not a valid image file')
                        print(e)
                        bad_images.append(f_path)
                else:
                    print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
        else:
            print('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')
    return bad_images, bad_ext


def show_check_images():
    '''
    调用函数输出结果
    '''
    source_dir = r'/home/mist/src/leaf_images'
    good_exts = ['jpg', 'jpeg']  # list of acceptable extensions
    bad_file_list, bad_ext_list = check_images(source_dir, good_exts)
    if len(bad_file_list) != 0:
        print('improper image files are listed below')

    print(bad_file_list)
    print(bad_ext_list)


def rename_image(input_folder):
    '''
    修改图片格式
    '''
    for filename in os.listdir(input_folder):
        # 确保文件是图像文件
        if not filename.endswith('.jpg') and not filename.endswith('.png'):
            continue

        # 打开图像文件并转换为RGB模式
        with Image.open(os.path.join(input_folder, filename)) as im:
            im = im.convert('RGB')

            # 另存为JPEG格式
            new_filename = os.path.splitext(filename)[0] + '.jpeg'
            im.save(os.path.join(input_folder, new_filename), 'JPEG')

        # 删除原始图像文件
        os.remove(os.path.join(input_folder, filename))


def get_subdirectories(path):
    """
    获取给定路径下所有子目录的绝对路径
    """
    subdirectories = []
    for root, dirs, files in os.walk(path):
        for d in dirs:
            subdirectories.append(os.path.abspath(os.path.join(root, d)))
    return subdirectories


if __name__ == '__main__':

    dic_path = '/home/mist/src/leaf_images'
    dics = get_subdirectories(dic_path)
    for dic in dics:
        rename_image(dic)

