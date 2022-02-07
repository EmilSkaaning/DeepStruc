import os, sys
import numpy as np
from scipy import spatial
import shutil
from tqdm import tqdm

def remove_duplicates(data_base):
    src, dst = copy_files(data_base)

    """dir1 = f'{data_base}/xyz_raw'
    files1 = sorted(os.listdir(dir1))

    print(f'{len(files1)} files are being checked.')
    moved_files, identical_files = 0, 0
    pbar = tqdm(total=len(files1))
    for idx1 in range(len(files1)):
        clone = False
        file1 = files1[idx1]
        dat1 = np.loadtxt(dir1+'/'+file1, skiprows=1, dtype=np.str)
        dat1 = np.array(dat1.T[1:], dtype=np.float)
        dist_mat1 = spatial.distance.cdist(dat1.T, dat1.T, 'euclidean')
        dist_mat1 = np.sort(dist_mat1.reshape(-1))
        pbar.update()
        for idx2 in range(idx1+1, len(files1)):
            file2 = files1[idx2]

            dat2 = np.loadtxt(dir1+'/'+file2, skiprows=1, dtype=np.str)
            dat2 = np.array(dat2.T[1:], dtype=np.float)

            if dat1.shape == dat2.shape:
                dist_mat2 = spatial.distance.cdist(dat2.T, dat2.T, 'euclidean')
                dist_mat2 = np.sort(dist_mat2.reshape(-1))

                val = np.array_equal(dist_mat1, dist_mat2)

                if val == True:
                    clone = True
                    break

        if clone:
            identical_files+=1
        else:
            moved_files += 1
            shutil.copy2(f'{dir1}/{file1}', f'{data_base}/xyz_unique/{file1}')

    pbar.close()"""

    files1 = sorted(os.listdir(dst))
    remove_list = []

    pbar = tqdm(total=len(files1))
    for idx1 in range(len(files1)):
        file1 = files1[idx1]
        dat1 = np.loadtxt(dst + '/' + file1, skiprows=1, dtype=np.str)
        dat1 = np.array(dat1.T[1:], dtype=np.float)
        dist_mat1 = spatial.distance.cdist(dat1.T, dat1.T, 'euclidean')
        dist_mat1 = np.sort(dist_mat1.reshape(-1))
        pbar.update()
        for idx2 in range(idx1, len(files1)):
            file2 = files1[idx2]

            if file1 == file2:
                continue
            elif file2 in remove_list or file1 in remove_list:
                continue

            dat2 = np.loadtxt(dst + '/' + file2, skiprows=1, dtype=np.str)
            dat2 = np.array(dat2.T[1:], dtype=np.float)

            if dat1.shape == dat2.shape:
                dist_mat2 = spatial.distance.cdist(dat2.T, dat2.T, 'euclidean')
                dist_mat2 = np.sort(dist_mat2.reshape(-1))

                val = np.array_equal(dist_mat1, dist_mat2)

                if val == True:
                    remove_list.append(file1)
                    os.remove(dst + '/' + file1)

    print(f'{len(os.listdir(dst))} were moved and {len(remove_list)} were clones.')

    return None


def copy_files(data_base):
    dst = f'{data_base}/xyz_unique'
    src = f'{data_base}/xyz_raw'
    try:
        os.makedirs(dst)
    except FileExistsError:
        pass

    xyz_files = os.listdir(src)
    pbar = tqdm(total=len(xyz_files))
    for file in xyz_files:
        shutil.copyfile(src+'/'+file,
                        dst+'/'+file)
        pbar.update()
    pbar.close()
    return src, dst
