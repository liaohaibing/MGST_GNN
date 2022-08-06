from pyproj import Proj
import numpy as np
from os.path import join as pjoin
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import math
import sklearn.datasets as datasets
import os
import shutil

ori_path='D:/项目资料/环保项目/机器学习应用/秸秆焚烧识别/positive'
dest_path='D:/项目资料/环保项目/机器学习应用/秸秆焚烧识别/unlabel_imgs'
for root, dirs, files in os.walk(ori_path):
    for file in files:
        temp=os.path.splitext(file)[0]
        if temp.split('_')[-1] == 'unmark':
            old_path = os.path.join(ori_path, file)
            new_path = os.path.join(dest_path, file)
            shutil.copyfile(old_path, new_path)


            ###--求多点外接多边形（Delaunay三角网）非凸-----# Computing the alpha shape
    # 通过这里的alpha阈值，可以得到不同的外接多边形。阈值选的不好，可能得不到外接多边形。比如选的太小。
def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)

    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle

    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        print('circum_r', circum_r)

        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def get_bottom_point(points):
    """
    返回points中纵坐标最小的点的索引，如果有多个纵坐标最小的点则返回其中横坐标最小的那个
    :param points:
    :return:
    """
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (
                points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i
    return min_index


def sort_polar_angle_cos(points, center_point):
    """
    按照与中心点的极角进行排序，使用的是余弦的方法
    :param points: 需要排序的点
    :param center_point: 中心点
    :return:
    """
    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i]
        point = [point_[0] - center_point[0], point_[1] - center_point[1]]
        rank.append(i)
        norm_value = math.sqrt(point[0] * point[0] + point[1] * point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)

    for i in range(0, n - 1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index - 1] or (
                    cos_value[index] == cos_value[index - 1] and norm_list[index] > norm_list[index - 1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index - 1]
                rank[index] = rank[index - 1]
                norm_list[index] = norm_list[index - 1]
                cos_value[index - 1] = temp
                rank[index - 1] = temp_rank
                norm_list[index - 1] = temp_norm
                index = index - 1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])

    return sorted_points


def vector_angle(vector):
    """
    返回一个向量与向量 [1, 0]之间的夹角， 这个夹角是指从[1, 0]沿逆时针方向旋转多少度能到达这个向量
    :param vector:
    :return:
    """
    norm_ = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    if norm_ == 0:
        return 0

    angle = math.acos(vector[0] / norm_)
    if vector[1] >= 0:
        return angle
    else:
        return 2 * math.pi - angle


def coss_multi(v1, v2):
    """
    计算两个向量的叉乘
    :param v1:
    :param v2:
    :return:
    """
    return v1[0] * v2[1] - v1[1] * v2[0]


def graham_scan(points):
    # print("Graham扫描法计算凸包")
    bottom_index = get_bottom_point(points)
    bottom_point = points.pop(bottom_index)
    sorted_points = sort_polar_angle_cos(points, bottom_point)

    m = len(sorted_points)
    if m < 2:
        print("点的数量过少，无法构成凸包")
        return

    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])

    for i in range(2, m):
        length = len(stack)
        top = stack[length - 1]
        next_top = stack[length - 2]
        v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
        v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        while coss_multi(v1, v2) >= 0:
            stack.pop()
            length = len(stack)
            top = stack[length - 1]
            next_top = stack[length - 2]
            v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        stack.append(sorted_points[i])

    return stack


if __name__ == "__main__":

    # points = [(319, 320), (125, 198), (250, 366), (129, 182), (262, 375), (235, 344), (248, 369), (367, 261),
    #           (196, 307), (163, 286)]
    # # points2 = [(1.1, 3.6),(2.1, 5.4), (2.5, 1.8),(3.3, 3.98), (4.8, 6.2), (4.3, 4.1), (4.2, 2.4), (5.9, 3.5),
    # #            (6.2, 5.3),(6.1, 2.56),(7.4, 3.7),(7.1, 4.3),(7, 4.1)]
    #
    # points2 = [[1.1, 3.6],
    #            [2.1, 5.4],
    #            [2.5, 1.8],
    #            [3.3, 3.98],
    #            [4.8, 6.2],
    #            [4.3, 4.1],
    #            [4.2, 2.4],
    #            [5.9, 3.5],
    #            [6.2, 5.3],
    #            [6.1, 2.56],
    #            [7.4, 3.7],
    #            [7.1, 4.3],
    #            [7, 4.1]]
    #
    # pts = np.array(points).astype(np.int32)
    # points = pts
    # pts2 = np.array(points2).astype(np.int32)
    # points2 = pts2
    #
    # edges = alpha_shape(points2, alpha=200, only_outer=True)
    # print('edges', edges)
    #
    # # Plotting the output
    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.axis('equal')
    # plt.plot(points2[:, 0], points2[:, 1], '.', color='b')
    # for i, j in edges:
    #     # print(points[[i, j], 0], points[[i, j], 1])
    #     ax.plot(points2[[i, j], 0], points2[[i, j], 1], color='red')
    #     pass
    # # ax.invert_yaxis()
    # plt.show()

    # points = [[1.1, 3.6],
    #           [2.1, 5.4],
    #           [2.5, 1.8],
    #           [3.3, 3.98],
    #           [4.8, 6.2],
    #           [4.3, 4.1],
    #           [4.2, 2.4],
    #           [5.9, 3.5],
    #           [6.2, 5.3],
    #           [6.1, 2.56],
    #           [7.4, 3.7],
    #           [7.1, 4.3],
    #           [7, 4.1]]

    # for point in points:
    #     plt.scatter(point[0], point[1], marker='o', c='y')
    #
    # result = graham_scan(points)
    #
    # length = len(result)
    # for i in range(0, length - 1):
    #     plt.plot([result[i][0], result[i + 1][0]], [result[i][1], result[i + 1][1]], c='r')
    # plt.plot([result[0][0], result[length - 1][0]], [result[0][1], result[length - 1][1]], c='r')
    #
    # print('result', result)
    # plt.show()

    a ={1000.0: ([119.17, 119.105, 119.019, 118.911, 118.793, 118.674], [34.85, 34.856, 34.864, 34.873, 34.881, 34.888]),
        100.0: ([119.17, 119.153, 119.12, 119.07, 119.01, 118.941], [34.85, 34.868, 34.887, 34.907, 34.925, 34.941]),
        500.0: ([119.17, 119.142, 119.087, 119.002, 118.897, 118.781], [34.85, 34.871, 34.896, 34.925, 34.95, 34.967])}

    b1 = np.array(a[1000.0])
    b2 = np.array(a[100.0])
    b3 = np.array(a[500.0])
    c2 = np.hstack((b1,b2,b3))
    c3=c2.transpose().tolist()


    iris = datasets.load_iris()
    data = iris.data
    points_ = data[:, 0:2]
    points__ = points_[0:50, :]
    points = points__.tolist()

    temp_index = 0
    for point in points:
        plt.scatter(point[0], point[1], marker='o', c='y')
        index_str = str(temp_index)
        plt.annotate(index_str, (point[0], point[1]))
        temp_index = temp_index + 1

    result = graham_scan(points)
    print(result)
    length = len(result)
    for i in range(0, length - 1):
        plt.plot([result[i][0], result[i + 1][0]], [result[i][1], result[i + 1][1]], c='r')
    plt.plot([result[0][0], result[length - 1][0]], [result[0][1], result[length - 1][1]], c='r')

    plt.show()






