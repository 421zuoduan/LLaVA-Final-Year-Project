import pickle

def read_pkl_file(file_path):
    """
    读取指定路径的pkl文件并输出内容

    Args:
        file_path (str): pkl文件的路径
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(data)
    except FileNotFoundError:
        print(f"文件未找到：{file_path}")
    except pickle.UnpicklingError:
        print(f"文件解码错误：{file_path}")
    except Exception as e:
        print(f"发生错误：{e}")

# 示例
file_path = "playground/data/eval/pope/entropy_values.pkl"  # 替换为你的pkl文件路径
read_pkl_file(file_path)