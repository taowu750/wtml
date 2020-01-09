#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
处理机器学习数据的模块，主要处理 .mldata 文件

.mldata 是自定义文件格式。它包含数据描述头和数据体。数据描述头是 json 格式，用来描述数据的构成，包含
样本集的名称（可选）、分隔符（可选，默认为","）、特征组、标记等。当文件的第一个非空行是 "header" 时，
表示接下来是数据描述头，否则将没有数据描述头。
其中特征组是一个数组，包含所有特征的定义。每个特征又包含名称（可选）、类型（nominal标称；ordinal有序；numerical
数值）、键值对数组（这个属性只有当类型为nominal或ordinal是才有效）。标记和特征的定义相同。
数据体类似.csv格式，全都是数字和分隔符组成的矩阵。数据体和数据描述头隔着一个空行。数据体的上方还可以写上版本信息，
比如最初版本的数据和处理后的数据。每个版本的数据都必须是同一份数据的不同表示。第一份版本的数据必须和
数据描述头中的定义完全相同。
"""

import json
import os.path as pth
from collections import OrderedDict
from collections.abc import Iterable
from functools import reduce
from io import StringIO

import numpy as np
from numpy import ndarray


# noinspection PyAttributeOutsideInit
class Prop:
    """
    属性，包含名称（可选）、类型（nominal标称；ordinal有序；numerical数值）、键值对数组（这个
    属性只有当类型为nominal或ordinal是才有效）。
    """

    __categories = {'nominal', 'ordinal', 'numerical'}

    def __init__(self, category: str, *, name: str = None, kvs: dict = None):
        """
        初始化 Prop。类型 category 值必须为 nominal（标称）、ordinal（有序），numerical（数值）
        中的一个。当类型为 nominal 或 ordinal 时，kvs 参数必须提供，它是一个 dict，键表示属性值
        的名称，值表示属性值。
        Prop 的 kvs 包含了从 key 到 value 和从 value 到 key 的映射。

        :param category: 类型
        :param name: 名称
        :param kvs: 属性名称-值键值对
        """

        if category not in self.__categories:
            raise ValueError('category must be nominal, ordered, one of the values')
        if (category == 'nominal' or category == 'ordinal') and kvs is None:
            raise ValueError('nominal prop or ordinal prop must provide "kvs"')

        self.category = category
        self.name = name
        self._kvs = {}
        self._init_kvs = kvs
        if kvs:
            for k, v in kvs.items():
                self._kvs[k] = v
                self._kvs[v] = k

    def __str__(self):
        return 'Prop {category=' + self.category + \
               ', name=' + self.name + \
               ', kvs=' + str(self._init_kvs) + '}'

    # TODO: 实现 __getitem__ 还必须实现 __contains__，否则使用 in 时会卡住
    def __contains__(self, key):
        return key in self._kvs

    def __getitem__(self, key):
        return self.kv(key)

    def kv(self, key: str or int, default=None) -> str or int:
        """
        由属性值名返回属性值，或由属性值返回属性值名。不存在或没有 kvs 返回 default。

        :param key: 属性值或属性值名
        :param default: 属性值名或属性值
        :return: 不存在的情况下返回的默认值
        """

        return self._kvs.get(key, default)

    def keys(self):
        """
        返回属性的所有键组成的 tuple。

        :return: 属性的所有键组成的 tuple
        """

        if not hasattr(self, '_keys'):
            self._keys = tuple((k for k in self._kvs.keys() if isinstance(k, str)))

        return self._keys

    def values(self):
        """
        返回属性的所有值组成的 tuple。

        :return: 属性的所有值组成的 tuple
        """

        if not hasattr(self, '_values'):
            self._values = tuple((v for v in self._kvs.keys() if isinstance(v, int)))

        return self._values


# noinspection PyAttributeOutsideInit,PyTypeChecker
class Props:
    """
    属性组定义。我们可以通过名称或者标号来访问某个属性。
    """

    def __init__(self, props: Iterable):
        """
        使用 props 进行初始化，props 要么是包含一系列属性 dict 的 Iterable，要么是
        包含一系列 Prop 的 Iterable。

        :param props: 属性组
        """

        self._props = OrderedDict()
        self._ni_map = {}

        idx = 0
        for prop in props:
            if isinstance(prop, dict):
                if 'category' not in prop:
                    raise TypeError('prop must contains "category"')
                category = prop['category']
                name = prop['name'] if 'name' in prop else None
                kvs = prop['kvs'] if 'category' in prop else None

                attr = Prop(category, name=name, kvs=kvs)
                if name is not None:
                    self._ni_map[name] = idx
                    self._props[name] = attr
                self._ni_map[idx] = name
                self._props[idx] = attr
            else:
                name = prop.name
                if name is not None:
                    self._ni_map[name] = idx
                    self._props[name] = prop
                self._ni_map[idx] = name
                self._props[idx] = prop

            idx = idx + 1

    def __contains__(self, key: str or int) -> bool:
        return key in self._props

    def __getitem__(self, key: str or int) -> Prop:
        return self.get_prop(key)

    def __iter__(self):
        """
        此迭代器每次返回一个元组，包含有标号、名称和 Prop。
        """

        return ((i, self.ni(i), prop) for i, prop in self._props.items() if isinstance(i, int))

    def __str__(self):
        if not hasattr(self, '_str'):
            self._str = 'Props [\n\t' + reduce(lambda o, n: o + ',\n\t' + n,
                                               map(lambda prop: str(prop), self.props())) + '\n]'

        return self._str

    def get_prop(self, key: str or int, default=None) -> Prop:
        """
        获取属性，不存在返回 default。可以使用属性名称或标号访问属性。返回属性中包含
        type、name 和 kvs，其中 kvs 包含了从 key 到 value 和从 value 到 key 的映射。
        注意 name 可能为 None，kvs 在 type 为 numerical 的情况下也为 None。

        :param key: 属性名称或标号。
        :param default: 不存在属性的情况下返回的默认值
        :return: 属性 dict
        """

        return self._props.get(key, default)

    def ni(self, key: str or int, default=None) -> str or int:
        """
        如果 key 是标号，返回对应的 name；否追如果 key 是 name，则返回对应的标号。
        不存在返回 default。

        :param key: 标号或 name
        :param default: 不存在时返回的默认值
        :return: 对应的值
        """

        return self._ni_map.get(key, default)

    def is_nominal(self, key: str or int) -> bool:
        """
        判断指定属性是否是标称类型，可以使用名称或标号访问。如果 key 不存在抛出
        KeyError 异常。

        :param key: 属性名称或标号。
        :return: 是否为标称类型
        """

        return self.__obtain_prop(key).category == 'nominal'

    def is_ordinal(self, key: str or int) -> bool:
        """
        判断指定属性是否是有序类型，可以使用名称或标号访问。如果 key 不存在抛出
        KeyError 异常。

        :param key: 属性名称或标号。
        :return: 是否为有序类型
        """

        return self.__obtain_prop(key).category == 'ordinal'

    def is_numerical(self, key: str or int) -> bool:
        """
        判断指定属性是否是数值类型，可以使用名称或标号访问。如果 key 不存在抛出
        KeyError 异常。

        :param key: 属性名称或标号。
        :return: 是否为数值类型
        """
        return self.__obtain_prop(key).category == 'numerical'

    def category(self, key: str or int, default=None) -> str:
        """
        获取指定属性的类型，可以使用名称或标号访问。如果 key 不存在返回 default

        :param key: 属性名称或标号。
        :param default: 不存在时返回的默认值
        :return: 类型
        """

        prop = self.get_prop(key, default)

        return prop.category if prop is not None else None

    def kvs(self, key: str or int, default=None):
        """
        获取指定属性的 kvs，可以使用名称或标号访问。如果 key 不存在返回 default

        :param key: 属性名称或标号
        :param default: 不存在时返回的默认值
        :return: kvs
        """

        prop = self.get_prop(key, default)

        return prop._kvs if prop is not None else None

    def values(self, key: str or int, default=None) -> tuple:
        """
        获取指定属性的所有值，可以使用名称或标号访问。如果 key 不存在返回 default

        :param key: 属性名称或标号
        :param default: 不存在时返回的默认值
        :return: 指定属性的所有值组成的 tuple
        """

        if isinstance(key, str):
            key = self.ni(key)
            if key is None:
                return default

        return self.as_index_values().get(key, default)

    def names(self) -> tuple:
        """
        返回所有 name 组成的 tuple。

        :return: 所有 name 组成的 tuple
        """

        if not hasattr(self, '_names'):
            self._names = tuple((name for name in self._props.keys() if isinstance(name, str)))

        return self._names

    def indexes(self) -> tuple:
        """
        返回所有标号组成的 tuple。

        :return: 所有标号组成的 tuple
        """

        if not hasattr(self, '_indexes'):
            self._indexes = tuple((idx for idx in self._props.keys() if isinstance(idx, int)))

        return self._indexes

    def props(self) -> tuple:
        """
        返回所有 prop 组成的 tuple。

        :return: 所有 prop 组成的 tuple
        """

        if not hasattr(self, '_ps'):
            self._ps = tuple((prop for idx, prop in self._props.items() if isinstance(idx, int)))

        return self._ps

    def as_index_values(self) -> dict:
        """
        返回一个标号-属性值 dict，键是属性的标号，值是一个包含该属性所有值的 tuple。

        :return: 标号-属性值 dict
        """

        if not hasattr(self, '_index_values'):
            self._index_values = {k: tuple((v for v in kvs.values())) for k, kvs in self._props.items()
                                  if isinstance(k, int)}

        return self._index_values

    def __obtain_prop(self, key: str or int) -> Prop:
        prop = self[key]
        if prop is None:
            raise KeyError('key "%s" do not exist' % key)

        return prop


class MLData:
    """
    处理和生成 .mldata 文件的类。
    它还可以处理 .txt 和 .csv 的文件，只要数据符合 .mldata 或 .csv 的规范即可。

    每个 data 都包含 X 和 Y 的数据。如果 features 和 label 都存在或都不存在就将数据集分成 X 和 Y（只有一列视为 Y）；
    存在一个则表示数据只有一个，另一个为 None。
    你也可以使用 merge 将 X 和 Y 合并为 X，或使用 split 将 X 切分为 X 和 Y。
    """

    def __init__(self, *, path: str = None, content: str = None, encoding: str = 'utf-8'):
        """
        进行初始化。注意，如果提供了 path，那么 content 将会被忽略掉。

        :param path: 需要读取的文件路径。必须是 txt、csv 或 mldata 格式的文件，否则会抛出异常
        :param content: 需要解析的内容。当提供了 path 时，此项会被忽略。
        :param encoding: 解析时的编码格式，默认为 utf-8
        """

        self.name = 'data'
        self.features = None
        self.label = None

        self._data = {}
        self._path = path
        self._delimiter = ','

        if path is not None:
            if not pth.isfile(path):
                raise IOError('file represented by path does not exist')
            if not path.endswith(('.mldata', '.txt', 'csv')):
                raise IOError('the path must be a mldata, txt, or csv file')

            with open(path, 'r', encoding=encoding) as f:
                self.__process(f)
        elif content is not None:
            self.__process(StringIO(content))

    def load(self):
        pass

    def write(self):
        pass

    def get_data(self, *, version: [str, int] = 0, default=None) -> dict:
        """
        获取指定版本的数据。未指定版本则选择数据集中首个版本的数据。

        :param version: 版本，可以是名称也可以是标号（即从 0 开始的第几个版本）
        :param default: 不存在时返回的默认值
        :return: 数据，包含 X 和 Y
        """

        return self._data.get(version, default)

    def add_data(self, data: ndarray or str, *, version: str = None, merge_to: str or int = None):
        pass

    def merge(self, *versions):
        """
        将指定的 versions 版本数据的 X 和 Y 合并为 X（如果 Y 不为 None）。如果 versions 为空，表示合并所有。

        :param versions: 指定的版本
        """

        if len(versions) == 0:
            for version, data in self._data.items():
                if isinstance(version, int) and data['Y']:
                    data['Y'] = data['Y'].reshape((data['Y'].shape[0], 1))
                    data['X'] = np.hstack((data['X'], data['Y'])) if data['X'] else data['Y']
                    data['Y'] = None
        else:
            for version in versions:
                if version in self._data and self._data[version]['Y']:
                    data = self._data[version]
                    data['Y'] = data['Y'].reshape((data['Y'].shape[0], 1))
                    data['X'] = np.hstack((data['X'], data['Y'])) if data['X'] else data['Y']
                    data['Y'] = None

    def split(self, *versions):
        """
        将指定 versions 版本数据的 X 分割为 X 和 Y（如果 X 不为 None，X 不止一列 且 Y 为 None）。
        如果 versions 为空，表示分割所有。

        :param versions: 指定的版本
        """

        if len(versions) == 0:
            for version, data in self._data.items():
                if isinstance(version, int) and data['X'] and data['X'].shape[1] > 1 and data['Y'] is None:
                    data['X'], data['Y'] = data['X'][:, :-1], data['X'][:, -1]
        else:
            for version in versions:
                if version in self._data and self._data[version]['X'] and self._data[version]['X'].shape[1] > 1 \
                        and self._data[version]['Y'] is None:
                    data = self._data[version]
                    data['X'], data['Y'] = data['X'][:, :-1], data['X'][:, -1]

    def __process(self, f):
        line = self.__skip_space(f)
        if line == '':
            raise IOError('file represented by path is empty')

        line = line.strip()
        if line.startswith('{'):
            # 处理数据头
            header = line
            while line != '':
                line = f.readline().strip()
                header = header + line
            header = json.loads(header)
            if 'name' in header:
                self.name = header['name']
            if 'delimiter' in header:
                self._delimiter = header['delimiter']
            if 'features' in header:
                self.features = Props(header['features'])
            if 'label' in header:
                label = header['label']
                self.label = Prop(label['category'], name=label.get('name', None), kvs=label.get('kvs', None))

        line = self.__skip_space(f)
        if line != '':
            # 如果数据体存在
            version_id = 0
            while True:
                # 读取所有版本的数据
                line = line.strip()
                data = None
                version = version_id
                if line.startswith('version='):
                    version = line[len('version='):]
                else:
                    data = np.asarray(tuple(map(lambda x: float(x), filter(lambda s: s.strip() != '',
                                                                           line.split(self._delimiter)))))
                line = f.readline().strip()
                while line != '':
                    nums = np.asarray(tuple(map(lambda x: float(x), filter(lambda s: s.strip() != '',
                                                                           line.split(self._delimiter)))))
                    if data is None:
                        data = nums
                    else:
                        data = np.vstack((data, nums))
                    line = f.readline().strip()
                # 将 data 转化为 X 和 Y
                # 如果 features 和 label 都存在或都不存在就将数据集分成 X 和 Y（只有一列视为 Y）
                # 只存在一个则表示数据只有一个，另一个为 None
                dat = {}
                if (self.features and self.label) or \
                        (self.features is None and self.label is None):
                    if data.shape[1] > 1:
                        dat['X'] = data[:, :-1]
                        dat['Y'] = data[:, -1]
                    else:
                        dat['X'] = None
                        dat['Y'] = data.ravel()
                elif self.features:
                    dat['X'] = data
                    dat['Y'] = None
                else:
                    dat['X'] = None
                    dat['Y'] = data.ravel()
                self._data[version] = dat
                self._data[version_id] = dat
                version_id = version_id + 1

                line = self.__skip_space(f)
                if line == '':
                    break

    @staticmethod
    def __skip_space(f):
        line = f.readline()
        while line.isspace():
            line = f.readline()

        return line


if __name__ == '__main__':
    p = MLData(path='watermelon2-0.mldata')
    print(p.features, '\n')
    print(p.label, '\n')
    print(p.get_data()['X'], '\n')
    print(p.get_data()['Y'], '\n')
