# shared_embedding.py
import multiprocessing.shared_memory
import pickle
import numpy as np
import os

def create_shared_embedding(file_path, embedding_name="my_embedding"):
    """
    从 pickle 文件加载 embedding 数据，并将其放入共享内存中。

    Args:
        file_path (str): pickle 文件的路径。
        embedding_name (str): 共享内存的名称。

    Returns:
        tuple: 一个包含共享内存名称、shape 和 dtype 的元组。
    """
    try:
        with open(file_path, 'rb') as f:
            embedding = pickle.load(f)
            # 将字典转换为 numpy 数组
            embedding_array = np.array(list(embedding.values()))
            embedding_shape = embedding_array.shape
            embedding_dtype = embedding_array.dtype

            # 计算共享内存的大小
            itemsize = np.dtype(embedding_dtype).itemsize
            nbytes = embedding_array.size * itemsize

            # 创建共享内存
            shared_memory = multiprocessing.shared_memory.SharedMemory(create=True, size=nbytes, name=embedding_name)

            # 创建 numpy 数组的副本到共享内存中
            shared_array = np.ndarray(embedding_shape, dtype=embedding_dtype, buffer=shared_memory.buf)
            shared_array[:] = embedding_array[:]

            print(f"Embedding data loaded from {file_path} and placed in shared memory '{embedding_name}'.")

            return embedding_name, embedding_shape, embedding_dtype, list(embedding.keys())

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading embedding or creating shared memory: {e}")
        return None

def attach_to_shared_embedding(embedding_name, embedding_shape, embedding_dtype):
    """
    连接到现有的共享内存，并返回 numpy 数组。

    Args:
        embedding_name (str): 共享内存的名称。
        embedding_shape (tuple): embedding 数据的 shape。
        embedding_dtype (dtype): embedding 数据的 dtype。

    Returns:
        numpy.ndarray: 一个 numpy 数组，指向共享内存。
    """
    try:
        existing_shm = multiprocessing.shared_memory.SharedMemory(name=embedding_name)
        shared_array = np.ndarray(embedding_shape, dtype=embedding_dtype, buffer=existing_shm.buf)
        return shared_array, existing_shm
    except Exception as e:
        print(f"Error attaching to shared memory: {e}")
        return None, None