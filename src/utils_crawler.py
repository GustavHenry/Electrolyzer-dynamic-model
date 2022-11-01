import requests
import pandas
import pandas as pd
import numpy
import numpy as np
import datetime
import os
import time
from coord_convert.transform import bd2wgs
from geopy.distance import geodesic
from math import sqrt
from shapely.geometry import Point, Polygon, MultiPolygon
from tqdm import tqdm
from retry import retry
import h3
from shapely import wkt
import geopandas as gpd
from time import sleep
import json

"""
    Within this .py file, several crawler function have been summaried and formatted for future use
    所有使用了api访问的内容，都应当注意ak、key的使用次数，百度每天为30000个，高德可以自己申请，总共有2000次
        本文件中只有get_poi_json会进行批量爬取数据，应当注意使用次数限制，其余方法问题应当不大
    推荐使用get_poi_json，结合amap_get_city_border提供的城市边界信息，可以快速便利整个城市内部的poi信息
        应当注意返回结果中有些是不合理的，需要关注查询结果，会有很多无关项，需要进行清晰，并且在最后还需要用uid进行去重
"""

H3_ADDRESS_6 = "h3_address_6"
H3_ADDRESS_8 = "h3_address_8"
LONGITUDE = "longitude"
LATITUDE = "latitude"

# 爬取第三方充电站（从百度爬）
def amap_get_city_border(keywords="上海"):
    """本函数使用高德地图，可以自动返回查询的行政区的总体边界，建议查询直辖市、省份级别的数据
        每次只返回当前行政区的边界信息，不返回下一级别行政区信息

    Args:
        keywords (str, optional): 需要查询的行政区名称. Defaults to '上海'.

    Returns:
        border_lat,border_lng：所有分形体聚合后边界的总体维度、经度坐标点信息，不区分分形体
        border_multi_polygon: list, 根据边界的分形体信息，每个坐标点数超过阈值的分形体都被记录在内，包含维度、经度信息，可以直接用于polygon
    """

    my_key = "cdb7ed1a8948392a19c40453aa3da9f5"  # 高德地图的key
    subdistrict = "0"  # 不需要返回下级行政区域
    extensions = "all"  # 需要返回边界
    url_amap = "https://restapi.amap.com/v3/config/district?keywords={}&subdistrict={}&extensions={}&key={}".format(
        keywords, subdistrict, extensions, my_key
    )  # 查询行政区域的边界，然后取经纬度最大值与最小值
    data = requests.get(url_amap)
    result = data.json()["districts"][0]

    border = result["polyline"]
    border_lat = []  # 所有边界的纬度
    border_lng = []  # 所有边界的经度
    border_multi_polygon = []
    if "|" in border:  # 如果存在分形体，就需要考虑多个多边形
        border = border.split("|")
        for idx in range(len(border)):
            coord_pairs = border[idx].split(";")
            if len(coord_pairs) > 69:  # 如果边界的点数量少于69个，认为可以忽略
                border_pair = []
                for idx_coord in range(len(coord_pairs)):
                    cur_lng, cur_lat = map(float, coord_pairs[idx_coord].split(","))
                    border_lat.append(cur_lat)
                    border_lng.append(cur_lng)
                    border_pair.append((cur_lng, cur_lat))
                border_multi_polygon.append(border_pair)
    else:  # 如果不存在多个分形体，那么就可以简单化考虑，直接给定一个多边形
        coord_pairs = border.split(";")
        if len(coord_pairs) > 10:  # 如果边界的点数量少于10个，认为可以忽略
            border_pair = []
            for idx_coord in range(len(coord_pairs)):
                cur_lng, cur_lat = map(float, coord_pairs[idx_coord].split(","))
                border_lat.append(cur_lat)
                border_lng.append(cur_lng)
                border_pair.append((cur_lng, cur_lat))
            border_multi_polygon.append(border_pair)
    print(
        "the number of polygons in {} is {}".format(keywords, len(border_multi_polygon))
    )

    return border_lat, border_lng, border_multi_polygon


def get_mesh_grid(border_lat, border_lng, border_multi_polygon, grid_steps=60):
    """本函数中需要输入一个城市所有的边界的坐标点，以及边界分形体边界列表
    之后计算得到覆盖整个城市所需要的点阵默认是将城市的横竖每个边分成60等分，这个可以调整
    最终返回在当前设定下城市内部的所有点阵，以及可以完全覆盖城市需要用到的半径
    """

    """能够包络上海市的矩形的对角线长度为187.77km"""
    lat_min = min(border_lat)
    lat_max = max(border_lat)
    lng_min = min(border_lng)
    lng_max = max(border_lng)
    # grid_steps = 60     # 应当注意，等分之后格子横、竖方向的边长是不一样的
    # 因此应该选取长边进行计算
    """
        当上海的经纬度极限划分为：
            横竖20等分，横竖两点之间的距离分别为6985、7012米
                完全覆盖格子的圆半径约为4948.8米，上海范围内有162个点
            横竖30等分，横竖两点之间的距离分别为4576、4594米
                完全覆盖格子的圆的半径约为3242米，大致为h3 分辨率6，上海范围内有372个点
                充电站最密集的区域里面，一个圆内部可能超过100个，内环宜山路交叉口就超过了
            横竖50等分，横竖两点之间的距离分别为2708、2719米
                完全覆盖格子的圆的半径约为1918.9米，上海范围内有1057个点
                内环宜山路交叉口有83个站
    """
    lat_array = np.linspace(lat_min, lat_max, grid_steps)
    lng_array = np.linspace(lng_min, lng_max, grid_steps)

    lat_array, lng_array = np.meshgrid(lat_array, lng_array)
    edge1 = geodesic(
        (lat_array[0, 0], lng_array[0, 0]), (lat_array[0, 1], lng_array[0, 1])
    ).m
    edge2 = geodesic(
        (lat_array[0, 0], lng_array[0, 0]), (lat_array[1, 0], lng_array[1, 0])
    ).m
    request_radius = sqrt((edge1 / 2) ** 2 + (edge2 / 2) ** 2)
    polygons = []
    for polygon_border in border_multi_polygon:
        p = Polygon(polygon_border)
        polygons.append(p)
    polygons = MultiPolygon(polygons)
    request_points = []
    for i in range(grid_steps):
        for j in range(grid_steps):
            if polygons.contains(Point(lng_array[i, j], lat_array[i, j])):
                request_points.append(str(lat_array[i, j]) + "," + str(lng_array[i, j]))
    return request_points, request_radius


def name_split(name: str):
    """输入一个文本，然后输出修正后的小区名称
        主要是针对x号楼、东区等

    Args:
        name (str): 家充安装信息中的安装地址

    Returns:
        _type_: 修改后的小区名
    """
    stream = str(name)
    replacer_list = ["中国", "·"]
    for replacer in replacer_list:
        if replacer in stream:
            stream = stream.replace(replacer, "")
    spliter_list = ["中区", "西区", "东区", "北区", "南区"]
    for spliter in spliter_list:
        if spliter in stream:
            stream = stream.split(spliter)[0]
    spliter_list = ["-", "("]
    for spliter in spliter_list:
        if spliter in stream:
            stream = stream.split(spliter)[0]
    spliter_list = []
    for idx in range(500, -1, -1):
        spliter_list.append(str(idx) + "号楼")
    for spliter in spliter_list:
        if spliter in stream:
            stream = stream.split(spliter)[0]
    # spliter_list = []
    # for idx in range(500,-1,-1):
    #     spliter_list.append(str(idx) + '号院')
    # for spliter in spliter_list:
    #     if spliter in stream:
    #         stream = stream.split(spliter)[0]
    spliter_list = []
    for idx in range(150, -1, -1):
        spliter_list.append(str(idx) + "排")
    for spliter in spliter_list:
        if spliter in stream:
            stream = stream.split(spliter)[0]
    num_dic = {
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }
    spliter_list = []
    for idx in range(
        1,
        10,
    ):
        spliter_list.append(num_dic[str(idx)] + "区")
    for spliter in spliter_list:
        if spliter in stream:
            stream = stream.split(spliter)[0]
    return stream


def check_and_return(cur_json, key):
    """用于网页搜索的结果解析，如果没有这个字段，就返回空

    Args:
        cur_json (json): 网页搜索的返回结果，分别是'result','detail'
        key (str): 需要检索的键值

    Returns:
        str: 如果查询得到就返回文本格式的结果，如果查询不到就返回空值
    """
    if key in cur_json:
        return str(cur_json[key])
    else:
        return


@retry((ValueError, TypeError), tries=20, delay=0.1, backoff=2, max_delay=1)
def get_poi_json(coord, radius, query="充电站", tag="交通设施"):
    """根据给定的圆心位置、半径，得到范围内所有的poi信息，得到所有结果后还需要进行查重
        百度poi查询的标准文档：https://lbsyun.baidu.com/index.php?title=lbscloud/poitags

    Args:
        coord: 搜索点的纬度、经度信息：39.55245684745763,116.30951862711865
        radius: 圆形搜索的半径，单位为m
        query: 搜索的关键字，颗粒度更细
        tag: 搜索的大类

    Returns:
        total: 本地查询的poi数量，因为需要看总量是否超过100个，他每次查询总数大概最多只能返回100，因此需要把总量限制到80一下会比较稳妥
        results_json: 返回的结果的json文件，没有直接使用，只是作为数据结构保留，查看是否有问题
        results_poi: 返回结果的pois列表，应该已经包含了所能够获得的所有详细信息，每一条就是一个poi对象
    """

    """根据给定的圆心位置、半径，得到范围内所有的充电站详细信息"""
    my_ak = "5SdPec5VYBkz9kyxTfM54N5mcV5BvUBT"
    page_num = 0
    total = 0  # 直接访问搜索结果的总数，是能和最终的逐条查询结果对应上的
    results_poi = []  # 结果中的坐标为baidu坐标，需要转化成wgs
    results_json = []
    while 1:
        url_circle_poi = "https://api.map.baidu.com/place/v2/search?query={q}&tag={t}&location={coord}&extensions_adcode=true&coord_type=1&page_size=20&page_num={pn}&radius={r}&radius_limit=true&output=json&ak={ak}".format(
            q=query, t=tag, coord=coord, pn=page_num, r=radius, ak=my_ak
        )
        page_num += 1
        data = requests.get(url_circle_poi)
        try:
            re = data.json()
            total = re["total"]
            results_json.append(re)
            for poi in re["results"]:
                results_poi.append(poi)
            if len(re["results"]) < 20:
                break
        except:
            print("error in json process at " + coord + ",page: " + str(page_num))
    return total, results_json, results_poi


def get_city_pois_type(city, query="房地产", tag="写字楼", grid_step=30):
    """对指定城市内的指定类型的建筑进行搜索，并返回结果数据片段

    Args:
        city (str): 需要查询的城市名称
        query (str, optional): 需要查询的二级标签. Defaults to '房地产'.
        tag (str, optional): 需要查询的一级标签. Defaults to '写字楼'.
        grid_step (int, optional): 查询设定的网格刻度数量，如果目标密度较高，可以选择60，密度不高的话可以选择30. Defaults to 30.

    Returns:
        _type_: _description_
    """
    border_lat, border_lng, border_multi_polygon = amap_get_city_border(city)
    request_points, request_radius = get_mesh_grid(
        border_lat, border_lng, border_multi_polygon, grid_steps=grid_step
    )
    print("total points: {}, radius:{:2f}m".format(len(request_points), request_radius))
    df_res = pandas.DataFrame()
    print("start crawling data")
    for cur_coord in tqdm(request_points):
        total, results_json, results_poi = get_poi_json(
            coord=cur_coord, radius=request_radius, query=query, tag=tag
        )
        for idx in range(len(results_poi)):

            lng, lat = bd2wgs(
                results_poi[idx]["location"]["lng"], results_poi[idx]["location"]["lat"]
            )
            cur_res_df = pandas.DataFrame(
                {
                    "tag": [tag],
                    "query": [query],
                    "name": [results_poi[idx]["name"]],
                    "province": [results_poi[idx]["province"]],
                    "city": [results_poi[idx]["city"]],
                    "district": [results_poi[idx]["area"]],
                    "adcode": [check_and_return(results_poi[idx], "adcode")],
                    "baidu_uid": [results_poi[idx]["uid"]],
                    "wgs_lng": lng,
                    "wgs_lat": lat,
                    H3_ADDRESS_6: h3.geo_to_h3(lat, lng, 6),
                    "address": [check_and_return(results_poi[idx], "address")],
                }
            )
            df_res = df_res.append(cur_res_df, ignore_index=True)
    df_res["adjusted_name"] = df_res["name"].apply(lambda x: name_split(x))
    df_res = df_res.drop_duplicates(subset=["baidu_uid"])
    return df_res


def result_filter_3rd_charger(df):
    """如果进行poi查询查找的是充电站，则会有很多脏数据，需要手动清洗

    Args:
        df (pandas.Dataframe): 进行poi_type搜索的结果dataframe

    Returns:
        _type_: 给出筛选的结果
    """
    df = df[df["adjusted_name"].str.contains("充电")]
    df = df[~df["adjusted_name"].str.contains("怪兽")]
    # df = df[~df['adjusted_name'].str.contains('特斯拉')]
    return df


# province,city,district都在mainland的csv中筛选出来
@retry((ValueError, TypeError), tries=20, delay=0.1, backoff=2, max_delay=1)
def autoclaw():
    for pro in provinces:
        pro = random.choice(provinces)
        cities = list(set(C_S.loc[C_S["province"] == pro]["city"]))
        for city in cities:
            city = random.choice(cities)
            districts = list(set(C_S.loc[C_S["city"] == city]["district"]))
            for district in districts:
                district = random.choice(districts)
                if os.path.exists(
                    "CNC_240\{}_{}_{}_charger_site_list.csv".format(pro, city, district)
                ):
                    continue
                if pro != district:
                    df_g_3rd_charger = utils_crawler.get_city_pois_type(
                        city=district, query="充电站", tag="交通设施", grid_step=20
                    )
                else:
                    continue
                df_g_3rd_charger = result_filter_3rd_charger(df_g_3rd_charger)
                df_g_3rd_charger.to_csv(
                    "CNC_240\{}_{}_{}_charger_site_list.csv".format(pro, city, district)
                )
    return df_g_3rd_charger


def baidu_api_query(my_ak, query, tag, page_num):
    """本函数包含了针对百度地图最为基础的poi查询方法，可以完成基本的关键字搜索，并返回列表

    Args:
        有关query、tag，query是二级行业分类、tag为一级行业分类，百度的官方文档：https://lbsyun.baidu.com/index.php?title=lbscloud/poitags
        my_ak (str): 百度地图提供的搜索权限ak，即高德的key
        query (str): 二级行业分类标签
        tag (str): 一级行业分类标签
        page_num (str): 显示的页码，因为一次最多显示20条数据，因此超过的部分只能通过自动翻页来便利所有返回结果
    Sample:
        my_ak = '5SdPec5VYBkz9kyxTfM54N5mcV5BvUBT'
        query = '充电站'
        tag = '交通设施'

    Returns:
        json: 返回访问url结果的json文件，应当注意.json()过程可能会出错
    """
    u = "https://api.map.baidu.com/place/v2/search?query={query_key}&tag={tag_key}&region=上海&scope=2&page_size=20&page_num={page_num}&output=json&ak={ak}".format(
        query_key=query, tag_key=tag, page_num=page_num, ak=my_ak
    )
    # u = 'https://api.map.baidu.com/place/v2/search?query={query_key}&region=北京&scope=2&page_size=20&page_num={page_num}&output=json&ak={ak}'.format(query_key=query,page_num = page_num,ak = my_ak)
    return requests.get(u).json()


def address_2_url(address: str, my_ak: str) -> str:
    """使用百度接口的地址查询器URL，主要嵌套在函数内使用

    Args:
        address (str): 需要查询gps位置的地址，文本，测试使用的是北京家充的修改后的地址
        my_ak (str): 百度地图查询权限的ak

    Returns:
        str: 可以直接用于url查询的文本字段
    """
    return (
        "https://api.map.baidu.com/geocoding/v3/?address={}&output=json&ak={}".format(
            address, my_ak
        )
    )


@retry((ValueError, TypeError), tries=20, delay=0.1, backoff=2, max_delay=1)
def url_2_gps(url: str) -> str:
    """访问百度地图API，进行地理位置抓取，并且转换成wgs的坐标
        如果json过程错误，就返回空
    Args:
        url (str): 格式化后的查询url文本，应当使用address_2_url函数直接生成
    Returns:
        str: 格式化的gps文本，包含维度，经度；地址查询置信度，可以根据置信度进行筛选
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47"
    }  # 反扒机制
    data = requests.get(url, headers=headers)
    try:
        content = data.json()
        lng = content["result"]["location"]["lng"]
        lat = content["result"]["location"]["lat"]
        lng, lat = bd2wgs(lng, lat)  # 在geocoding用法中得到的结果需要进行转化，才能得到wgs84的坐标
        return str(lat) + "," + str(lng) + ";" + str(content["result"]["confidence"])
    except:
        print("error with json process")
        content = ""
        return


def add_gps(df: pandas.DataFrame, my_ak: str):
    """在dataframe中增加gps字段，主要用于批量化调用上面的地址查询函数
        目前主要是用在家充的数据中加入gps

        用时情况：
            1000 ：150s
            500  ：67s
            5    ：0.72s
            24974：3388s

        数据错误：
            大概每200条左右的数据，会有一个无法读到gps数据
        调用方法：
            add_gps(df_home,my_ak)
            可以便利列表中的所有家充数据，得到他们的gps与位置精确度，并且最后保存一个csv文件


    Args:
        df (pandas.DataFrame): 输入的dataframe，应当包含地址信息
        my_ak (str): 百度地图查询的权限ak
    """
    url_request_result_list = []
    print("Starting adding gps data to home charger records")
    for idx in tqdm(range(len(df))):
        url_request_result_list.append(
            url_2_gps(address_2_url(df.iloc[idx]["Address_adjusted"], my_ak=my_ak))
        )
    df["url_request_result"] = url_request_result_list
    # df['url_request_result'] = df['Address_adjusted'].apply(lambda x: url_2_gps(address_2_url(x,my_ak)))
    df = df.dropna(subset=["url_request_result"])
    df["wgs_gps"] = df["url_request_result"].apply(lambda x: x.split(";")[0])
    lats = []
    lngs = []
    wgs_lats = []
    wgs_lngs = []
    hex_6 = []
    hex_8 = []
    for idx in range(len(df)):
        lat, lng = map(float, df.iloc[idx]["wgs_gps"].split(","))
        lats.append(lat)
        lngs.append(lng)
        wgs_lng, wgs_lat = bd2wgs(lng, lat)
        wgs_lats.append(wgs_lat)
        wgs_lngs.append(wgs_lng)
        hex_6.append(h3.geo_to_h3(lat, lng, 6))
        hex_8.append(h3.geo_to_h3(lat, lng, 8))
    df["lat"] = lats
    df["lng"] = lngs
    df["wgs_lng"] = wgs_lngs
    df["wgs_lat"] = wgs_lats
    df["h3_address_6"] = hex_6
    df["h3_address_8"] = hex_8
    df["request_confidence"] = df["url_request_result"].apply(lambda x: x.split(";")[1])
    return df


def poi_2_formatted_df(tag, query, results_poi):
    """输入某地点所有搜索的poi结果，返回格式化后的数据片段

    Args:
        tag (str): 搜索过程的一级标签
        query (str): 搜索过程的二级标签
        results_poi (list): 搜索返回的json队列，每一个poi一个json的列表

    Returns:
        pandas.DataFrame: 格式化的结果数据片段
    """
    df = pandas.DataFrame()
    for idx in range(len(results_poi)):
        lng, lat = bd2wgs(
            results_poi[idx]["location"]["lng"], results_poi[idx]["location"]["lat"]
        )
        cur_res_df = pandas.DataFrame(
            {
                "tag": [tag],
                "query": [query],
                "name": [results_poi[idx]["name"]],
                "province": [results_poi[idx]["province"]],
                "city": [results_poi[idx]["city"]],
                "district": [results_poi[idx]["area"]],
                "adcode": [results_poi[idx]["adcode"]],
                "baidu_uid": [results_poi[idx]["uid"]],
                "wgs_lng": lng,
                "wgs_lat": lat,
            }
        )
        df = df.append(cur_res_df, ignore_index=True)
    return df


def hex_neighbor_search(distance_bar, df_target, df_sc_site, df_beijing_3rd_charger):
    """本函数输入需要匹配的目标地址信息以及超充、第三方充电站信息
        并且进行位置匹配，最终给出符合距离阈值的超充及第三方站点数量
        应当注意这里输入的数据应当是符合数据格式的，里面都需要有hex_6, wgs_lat, wgs_lng字段

    Args:
        distance_bar (float): 搜索范围
        df_target (pandas.DataFrame): 需要搜索充电桩的目标数据
        df_sc_site (pandas.DataFrame): 特斯拉超充站点的数据
        df_beijing_3rd_charger (pandas.DataFrame): 第三方站点的数据

    Returns:
        _type_: _description_
    """

    def cal_distance(row):
        """内部使用的计算两点间距的apply内函数"""
        cur_tesla_lat = row["wgs_lat"]
        cur_tesla_lng = row["wgs_lng"]
        distance = geodesic((cur_lat, cur_lng), (cur_tesla_lat, cur_tesla_lng))
        return distance

    list_counter_neighbor_tesla_sc = []
    list_counter_neighbor_3rd_site = []

    for idx in tqdm(range(len(df_target))):
        cur_hex = df_target.iloc[idx]["hex_6"]
        cur_lat = df_target.iloc[idx]["wgs_lat"]
        cur_lng = df_target.iloc[idx]["wgs_lng"]
        neighbor_tesla_sites = pandas.DataFrame()
        neighbor_3rd_sites = pandas.DataFrame()
        counter_neighbor_tesla_sc = 0
        counter_neighbor_3rd_site = 0
        if distance_bar < 2:
            k_rings_level = 1
        elif distance_bar < 4:
            k_rings_level = 2
        elif distance_bar < 6:
            k_rings_level = 3
        elif distance_bar < 12:
            k_rings_level = 4
        else:
            print("wrong distance bar{}".format(distance_bar))
            return

        for hex_idx in list(h3.k_ring(cur_hex, k_rings_level)):
            neighbor_tesla_sites = neighbor_tesla_sites.append(
                df_sc_site[df_sc_site["hex_6"] == hex_idx]
            )
            neighbor_3rd_sites = neighbor_3rd_sites.append(
                df_beijing_3rd_charger[df_beijing_3rd_charger["hex_6"] == hex_idx]
            )
        # 使用apply并行
        if len(neighbor_tesla_sites) > 0:
            neighbor_tesla_sites["distance"] = neighbor_tesla_sites.apply(
                lambda row: cal_distance(row), axis=1
            )
            counter_neighbor_tesla_sc += len(
                neighbor_tesla_sites[neighbor_tesla_sites["distance"] < distance_bar]
            )

        if len(neighbor_3rd_sites) > 0:
            neighbor_3rd_sites["distance"] = neighbor_3rd_sites.apply(
                lambda row: cal_distance(row), axis=1
            )
            counter_neighbor_3rd_site += len(
                neighbor_3rd_sites[neighbor_3rd_sites["distance"] < distance_bar]
            )

        list_counter_neighbor_tesla_sc.append(counter_neighbor_tesla_sc)
        list_counter_neighbor_3rd_site.append(counter_neighbor_3rd_site)
    return list_counter_neighbor_tesla_sc, list_counter_neighbor_3rd_site


def list_2_dataframe(
    cur_address, cur_lat, cur_lng, cur_ref_num, cur_jobid, cur_hex_8, cur_list: list
):
    """下面的函数需求的使用列表返回格式化df的函数

    Args:
        cur_address (str): 地址
        cur_lat (float): 经纬度
        cur_lng (float): 经纬度
        cur_jobid (str): 家充安装的工单ID，可以在系统中做唯一对应
        cur_hex_8 (str): 当前的hex-8编号
        cur_list (list): 进行百度搜索后返回的各种字段的集合

    Returns:
        pandas.Dataframe: 返回的格式化的df
    """
    cur_province = cur_list[0]
    cur_city = cur_list[1]
    cur_district = cur_list[2]
    cur_town = cur_list[3]
    cur_adcode = cur_list[4]
    cur_red_poi = cur_list[5]  # 可能会不准，仅供参考
    cur_ref_poi_code = cur_list[6]  # 可能会不准，仅供参考
    cur_community = cur_list[7]
    cur_community_tag = cur_list[8]
    cur_address_precise = cur_list[9]
    cur_address_confidence = cur_list[10]
    cur_address_comprehesion = cur_list[11]
    cur_dataframe = pandas.DataFrame(
        {
            "community": [cur_community],
            "address": [cur_address],
            "ref_poi": [cur_red_poi],
            "province": [cur_province],
            "city": [cur_city],
            "district": [cur_district],
            "town": [cur_town],
            "adcode": [cur_adcode],
            "lat": [cur_lat],
            "lng": [cur_lng],
            "reference_number": [cur_ref_num],
            "JobID": [cur_jobid],
            "hex_8": [cur_hex_8],
            "hex_6": [h3.h3_to_parent(cur_hex_8, 6)],
            "poi_type": [cur_community_tag],
            "ref_pot_code": [cur_ref_poi_code],
            "address_precise": [cur_address_precise],
            "address_confidence": [cur_address_confidence],
            "address_comprehesion": [cur_address_comprehesion],
        }
    )
    return cur_dataframe


def home_charger_add_gps(df_home_charger):
    """
    给家充的数据增加GPS信息
    """
    my_ak = "5SdPec5VYBkz9kyxTfM54N5mcV5BvUBT"  # 百度地图ak
    df_home_charger = Address_adjustor(df_home_charger)
    df_home_charger = add_gps(df_home_charger, my_ak)
    return df_home_charger


def home_charger_community_match(df_home_charger):
    """对家充的数据进行分析，使用百度的地址解析聚合服务，最终返回格式化的地址以及gps信息等
        主要用于解析家充安装的小区名称
        搜索500份的用时大概是84s

    Args:
        df_home_charger (pandas.Dataframe): 需要进行搜索的家充数据，关键字段是InstallationStreet，同时也需要具备百度地图的gps，之后要转化成wgs的格式

    Returns:
        _type_: 最后输出格式化之后的dataframe，不在原始的df上做修改
    """
    # 首先进行地址的清理
    my_ak = "5SdPec5VYBkz9kyxTfM54N5mcV5BvUBT"  # 百度地图ak
    # 可以开始结果的爬取
    df_res = pandas.DataFrame()
    for idx in tqdm(range(len(df_home_charger))):
        cur_address = df_home_charger.iloc[idx]["name"]
        # 原始数据中时没有这三个的，所以在上面需要加入
        cur_lat = df_home_charger.iloc[idx]["lat"]
        cur_lng = df_home_charger.iloc[idx]["lng"]
        cur_hex_8 = df_home_charger.iloc[idx]["hex_8"]
        cur_jobid = check_and_return(df_home_charger.iloc[idx], "JobID")
        cur_ref_num = cur_jobid = check_and_return(
            df_home_charger.iloc[idx], "reference_number"
        )
        url_address_analyzer = "https://api.map.baidu.com/address_analyzer/v1?address={add}&ak={ak}".format(
            add=cur_address, ak=my_ak
        )
        data = requests.get(url_address_analyzer)
        try:
            res_json = data.json()  # ['status', 'address', 'result', 'detail']
        except:
            print(
                "Error with json project at:{}, location:{}".format(
                    str(idx), cur_address
                )
            )
        if res_json["status"] == 0:
            check_result_list = [
                "province",
                "city",
                "county",
                "town",
                "town_code",
                "poi",
                "poi_code",
            ]  # poi可能是不准的，应该以detail里面的address_poi为准
            check_detail_list = [
                "address_poi",
                "poi_tag",
                "address_precise",
                "address_confidence",
                "address_comprehension",
            ]
            cur_res_list = []
            for key in check_result_list:
                cur_res_list.append(check_and_return(res_json["result"], key))

            if "detail" in res_json:  # 不确定没有返回结果时会怎么操作，先这么写着防止出错
                for key in check_detail_list:
                    cur_res_list.append(check_and_return(res_json["detail"], key))
            else:
                for key in check_detail_list:
                    cur_res_list.append("")

        cur_dataframe = list_2_dataframe(
            cur_address,
            cur_lat,
            cur_lng,
            cur_ref_num,
            cur_jobid,
            cur_hex_8,
            cur_list=cur_res_list,
        )
        df_res = df_res.append(cur_dataframe, ignore_index=True)
    return df_res


def request_target_aoi(target):
    """根据想要的目标爬取其网页上显示的AOI边界，应当注意的是，这里的目标应当尽量精确，否则容易偏移、出错
        名称中尽量包含地级市级别的文字，从而限定搜索的范围

    Args:
        target (str): 搜索的目标，尽量包含地级市

    Returns:
        np.arraay: 如果能够有结果，就返回一个边界的list，可以作为后续的处理起点，如果没有就返回一个空列表
    """

    def mercatortobd09(x, y):
        """
        墨卡托投影坐标转回bd09
        :param x:
        :param y:
        :return:
        """

        class LLT:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        def convertMC2LL(cB):
            # 百度墨卡托转回到百度经纬度纠正矩阵
            MCBAND = [12890594.86, 8362377.87, 5591021, 3481989.83, 1678043.12, 0]
            MC2LL = [
                [
                    1.410526172116255e-8,
                    0.00000898305509648872,
                    -1.9939833816331,
                    200.9824383106796,
                    -187.2403703815547,
                    91.6087516669843,
                    -23.38765649603339,
                    2.57121317296198,
                    -0.03801003308653,
                    17337981.2,
                ],
                [
                    -7.435856389565537e-9,
                    0.000008983055097726239,
                    -0.78625201886289,
                    96.32687599759846,
                    -1.85204757529826,
                    -59.36935905485877,
                    47.40033549296737,
                    -16.50741931063887,
                    2.28786674699375,
                    10260144.86,
                ],
                [
                    -3.030883460898826e-8,
                    0.00000898305509983578,
                    0.30071316287616,
                    59.74293618442277,
                    7.357984074871,
                    -25.38371002664745,
                    13.45380521110908,
                    -3.29883767235584,
                    0.32710905363475,
                    6856817.37,
                ],
                [
                    -1.981981304930552e-8,
                    0.000008983055099779535,
                    0.03278182852591,
                    40.31678527705744,
                    0.65659298677277,
                    -4.44255534477492,
                    0.85341911805263,
                    0.12923347998204,
                    -0.04625736007561,
                    4482777.06,
                ],
                [
                    3.09191371068437e-9,
                    0.000008983055096812155,
                    0.00006995724062,
                    23.10934304144901,
                    -0.00023663490511,
                    -0.6321817810242,
                    -0.00663494467273,
                    0.03430082397953,
                    -0.00466043876332,
                    2555164.4,
                ],
                [
                    2.890871144776878e-9,
                    0.000008983055095805407,
                    -3.068298e-8,
                    7.47137025468032,
                    -0.00000353937994,
                    -0.02145144861037,
                    -0.00001234426596,
                    0.00010322952773,
                    -0.00000323890364,
                    826088.5,
                ],
            ]

            def convertor(cC, cD):
                if cC == None or cD == None:
                    print("null")
                    return None
                T = cD[0] + cD[1] * abs(cC.x)
                cB = abs(cC.y) / cD[9]
                cE = (
                    cD[2]
                    + cD[3] * cB
                    + cD[4] * cB * cB
                    + cD[5] * cB * cB * cB
                    + cD[6] * cB * cB * cB * cB
                    + cD[7] * cB * cB * cB * cB * cB
                    + cD[8] * cB * cB * cB * cB * cB * cB
                )
                if cC.x < 0:
                    T = T * -1
                else:
                    T = T
                if cC.y < 0:
                    cE = cE * -1
                else:
                    cE = cE
                return [T, cE]

            cC = LLT(abs(cB.x), abs(cB.y))
            cE = None
            for cD in range(0, len(MCBAND), 1):
                if cC.y >= MCBAND[cD]:
                    cE = MC2LL[cD]
                    break
            T = convertor(cB, cE)
            return T

        baidut = LLT(x, y)
        return convertMC2LL(baidut)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47"
    }  # 反扒机制
    url = "https://map.baidu.com/?newmap=1&qt=s&da_src=searchBox.button&wd={}".format(
        target
    )
    response = requests.get(url, headers=headers)
    response.encoding = "utf-8"
    # return response.json()
    try:
        res_geo = response.json()["content"][0]["profile_geo"]
        res_aoi = res_geo.split("1-")[-1].split(";")[0]
        # 百度使用的可能是BD09mc坐标系，在官方文档中有提及 https://lbs.baidu.com/index.php?title=ios-locsdk/guide/addition-func/latlng-trans
        # 需要学习百度的各种坐标系 https://blog.csdn.net/sinat_41310868/article/details/115551276
        res_center = res_geo.split("|1-")[0].split("|")[-1]
        reslist = np.array(list(map(float, res_aoi.split(","))))
        reslist = reslist.reshape((len(reslist) // 2, 2))
        for idx in range(len(reslist)):
            reslist[idx, :] = mercatortobd09(reslist[idx, 0], reslist[idx, 1])
            reslist[idx, :] = bd2wgs(reslist[idx, 0], reslist[idx, 1])
        return reslist
    except:
        print("error with json process")
        return []


def aoi_offset(aoi, offset_m=50):
    """根据爬取的目标边界，进行外扩边缘计算，并最终返回polygon


    Args:
        aoi (list): 爬取的目标边界，如果是空的话，就会返回两个空的polygon
        offset_m (int, optional): 需要外扩的距离，单位为米，会自动转化成经纬度进行计算，略有误差. Defaults to 50.

    Returns:
        shapely.Polygon: 返回两个Polygon对象，分别是本身的AOI边界以及外扩后的边界
    """
    if not aoi.any():
        # 如果列表不存在，就返回两个空值
        ori_poly = Polygon(aoi)
        return ori_poly, ori_poly
    else:
        # 如果列表存在，则返回原始、经过offset的两个多边形
        ori_poly = Polygon(aoi)
        offset = offset_m / 100000  # 自动将米制转化成经纬度，只是模糊转化，并不是特别精确
        offset_poly = ori_poly.buffer(offset)
        return ori_poly, offset_poly


def Jia_office_crawler(prefix, city):
    """可以自动爬取某城市的甲级写字楼，并返回名称、地址、wgs坐标
        按照下面这个网站的标准来：http://bj.86office.com/office/_olid-1_isasc-1_pagenow-1.htm
    Args:
        prefix (str): 需要城市的名称标签
    """

    def name_2_gps(name, my_ak):
        url = address_2_url(name, my_ak=my_ak)
        res_gps = url_2_gps(url)
        if res_gps:
            res = res_gps.split(";")[0]
            wgs_lat, wgs_lng = map(float, res.split(","))
            return [wgs_lat, wgs_lng]
        else:
            return [0.0, 0.0]

    page_range = np.arange(1, 100)
    name_list = []
    address_list = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47"
    }  # 不知道是干嘛的，反正加上去有可能就对了
    print("extracting office building list from: {}".format(prefix))
    for page in tqdm(page_range):
        url = "http://{}.86office.com/office/_ocid-3_isasc-1_pagenow-{}.htm".format(
            prefix, page
        )
        data = requests.post(url, headers=headers)
        if data.status_code == 200:
            res = data.text
            res = res.split('<span class="m_floatleft">销售价格</span>\r\n')[1]
            res = res.split('" title="')[1:]
            idx = 5
            for idx in range(len(res) // 2):
                cur = res[idx * 2 - 1]
                name = cur.split('"')[0]
                address = cur.split(
                    "</div>\r\n                            <div class="
                )[0].split('<div class="ftlt_title font_xst">')[1]
                address = address.split("，")[0]
                address = address.split("(")[0]
                address = address.split("（")[0]
                name_list.append(name)
                address_list.append(address)
            if len(res) // 2 < 10:
                break
        # sleep(0.1)
    df_jia_office = pandas.DataFrame()
    df_jia_office["name"] = name_list
    df_jia_office["address"] = address_list
    df_jia_office = df_jia_office.drop_duplicates(subset=["name"])
    df_jia_office.reset_index(inplace=True)
    my_ak = "5SdPec5VYBkz9kyxTfM54N5mcV5BvUBT"
    df_jia_office[["latitude", "longitude"]] = df_jia_office["name"].apply(
        lambda x: pandas.Series(name_2_gps(city + x, my_ak=my_ak))
    )
    return df_jia_office


def polygon_list_2_str(polygon: list):

    """输入一个polygon的list<list>，返回一个string用于储存在pandas中

        暂时不需要这个东西了，未来如果需要用慧眼才需要



    Args:

        polygon (list): 包含坐标串的list



    Returns:

        str: 输出适合keplergl显示的文本

    """

    return ";".join(map(str, [",".join(map(str, x)) for x in polygon]))


@retry((ValueError, TypeError), tries=20, delay=0.1, backoff=2, max_delay=1)
def request_gps_polygon(lng, lat, contour_min):
    """根据爬取的办公楼信息，主要是需要使用经纬度信息以及等时图覆盖路程的信息
        调用方法：df_jingqu_all[[ 'contour_{}min'.format(contour_min[0]),
                'contour_{}min'.format(contour_min[1]),
                'contour_{}min'.format(contour_min[2]),]] = df_jingqu_all.apply(lambda row: pd.Series(request_gps_polygon(lng = row[LONGITUDE],lat=row[LATITUDE],contour_min=contour_min)),axis = 1)
        3000条数据跑了42min
    Args:
        lng (float): 目标点的经度
        lat (float): 目标点的纬度
        contour_min (int): 行驶时间长度
    Returns:
        Shapely.geometry.Polygon: 直接返回一个polygon对象，可以添加在pandas内部
    """
    res_seq = []
    polygon = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47"
    }  # 反扒机制
    MAPBOX_ACCESS_TOKEN = "pk.eyJ1Ijoic2NoZW4yMCIsImEiOiJja2c3YmJxdHYwNW1yMnNwZ3h4bmU0dHpkIn0.pwqkz9VKoj9zWG2BB7563g"  # dihan's token for mapbox
    url_isochrone = "https://api.mapbox.com/isochrone/v1/mapbox/driving/{}%2C{}?contours_minutes={}&polygons=true&denoise=1&access_token={}".format(
        lng, lat, ",".join(list(map(str, contour_min))), MAPBOX_ACCESS_TOKEN
    )
    data = requests.get(url_isochrone, headers=headers)

    res = data.json()
    if "features" in res:
        for contour_idx in range(len(contour_min)):
            polygon = res["features"][contour_idx]["geometry"]["coordinates"][0]
            res_seq.append(Polygon(polygon))
    else:
        for c in contour_min:
            res_seq.append(Polygon(polygon))
    res_seq.reverse()  # 给出的结果是倒序，所以需要再次倒序
    return res_seq


def target_sc_cover(df_target, df_sc, contour_mins, site_type="existing"):
    """根据给出的目标站点和超充站点数据，进行geopandas匹配计算，并且给出超充的覆盖情况
    应当注意的是，这里使用了geopandas的方法，因此对于目标的字段有明确要求，这里就是之前利用isochrone爬出的结果
    进行匹配之后，会给出结果，主要是当前的点位覆盖了多少个站点

    Args:
        df_target (pd.DataFrame): 目标数据，需要包含符合命名规则的polygon
        df_sc (pd.DataFrame): 超充站点，需要包含wgs经纬度信息
        contour_mins (list): 目标分钟数，应当是整数
        site_type (str, optional): 正在分析的类型，主要用于加入字段中，以方便进行区分. Defaults to 'existing'.

    Returns:
        pd.DataFrame: 在原有输入目标中加入相应字段后返回结果
    """

    df_sc["geometry"] = df_sc.apply(
        lambda row: Point(row[LONGITUDE], row[LATITUDE]), axis=1
    )
    df_sc_geo = gpd.GeoDataFrame(df_sc)
    df_target["target_id"] = range(len(df_target))
    for cmin in contour_mins:
        if isinstance(df_target.iloc[0]["contour_{}min".format(cmin)], str):
            df_target["contour_{}min".format(cmin)] = df_target[
                "contour_{}min".format(cmin)
            ].apply(wkt.loads)
    for cmin in contour_mins:
        df_target = df_target.rename(columns={"contour_{}min".format(cmin): "geometry"})

        df_target_geo = gpd.GeoDataFrame(df_target)
        df_res = df_target_geo.sjoin(df_sc_geo, how="left")
        df_res = pd.DataFrame(df_res)
        df_res = df_res.dropna(subset=["total_posts"])
        # return df_res
        df_count = df_res.value_counts(subset=["target_id"])
        df_count = df_count.reset_index()

        df_count.columns = ["target_id", site_type + "_sc_sites_{}min".format(cmin)]
        df_target = pd.merge(left=df_target, right=df_count, how="left", on="target_id")

        df_target = df_target.rename(
            columns={"geometry": "contour_{}min".format(cmin)}
        )  # 把名称再修改回去
        df_target = df_target.fillna(value=int(0))
    return df_target


def Address_adjustor(df: pandas.DataFrame) -> pandas.DataFrame:
    """本函数针对北京地区的家充安装数据，对其具体安装的位置进行初步文本解析
        删除会干扰搜索的词汇，并进行一些字段的替换

    Args:
        df (pandas.DataFrame): 输入的dataframe，主要是针对北京市的家充安装位置数据

    Returns:
        pandas.DataFrame: 对输入dataframe增加一个调整后的地址字段，然后再返回
    """
    df = df.dropna(subset=["name"])
    df["Address_adjusted"] = df["name"].apply(lambda x: x.replace("中国CN北京", ""))
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.split(",")[0])
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.split("，")[0])
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.split("（")[0])
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.split("#")[0])
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.replace("车位", ""))
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.replace("别墅", ""))
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.replace("自有", ""))
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.replace("地下", ""))
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.replace("CN", ""))
    df["Address_adjusted"] = df["Address_adjusted"].apply(lambda x: x.replace("中国", ""))
    df["Address_adjusted"] = df["Address_adjusted"].apply(
        lambda x: x.replace("北京北京", "北京")
    )
    return df


def pois_2_df(coord, pois):
    """将查询好的坐标和详细信息插入到dataframe里面，然后返回标准化的结果

    Args:
        coord (_type_): 实例为纬、经度：39.55245684745763,116.30951862711865
        pois (_type_): 根据百度pois查询得到的结果，包含很多字段

    Returns:
        df: pd.DataFrame: 当前查询pois的格式化数据返回结果
            包含字段：
                    name
                    address
                    province
                    city
                    area
                    uid
                    adcode
                    lat
                    lng
                    source_coord
    """

    def check_and_append(lst: list, poi, key):
        """判断字段是否存在，存在就插入表中，主要针对有的搜索结果会没有地址"""
        if key in poi.keys():
            lst.append(poi[key])
        else:
            lst.append("")
        return lst

    df = pandas.DataFrame()
    source_coord = [coord] * len(pois)
    names = []
    addresses = []
    lats = []
    lngs = []
    provs = []
    cities = []
    areas = []
    uids_baidu = []
    adcodes = []

    for poi in pois:
        names = check_and_append(names, poi, "name")
        addresses = check_and_append(addresses, poi, "address")
        provs = check_and_append(provs, poi, "province")
        cities = check_and_append(cities, poi, "city")
        areas = check_and_append(areas, poi, "area")
        uids_baidu = check_and_append(uids_baidu, poi, "uid")
        adcodes = check_and_append(adcodes, poi, "adcode")
        lats.append(poi["location"]["lat"])
        lngs.append(poi["location"]["lng"])

    df["names"] = names
    df["addresses"] = addresses
    df["provs"] = provs
    df["cities"] = cities
    df["areas"] = areas

    df["adcodes"] = adcodes
    df["lats"] = lats
    df["lngs"] = lngs
    df["uids_baidu"] = uids_baidu
    df["source_coord"] = source_coord
    return df


@retry((ValueError, TypeError), tries=20, delay=0.1, backoff=2, max_delay=1)
def fetch_community_list_from_LIANJIA(city="sh"):
    """可以从链家爬取某个城市的小区列表，但是可能没有房价信息，目前的输入只支持北京或者上海
        北京就输入'bj'，上海就输入'sh'
        本函数会不停retry，同时自带休眠反爬，而且可以反复进行缓存

    Returns:
        pd.DataFrame: 最终返回一个全的dataframe，数据已经经过了暂存
    """

    districts = {
        "bj": [
            "dongcheng",
            "xicheng",
            "chaoyang",
            "haidian",
            "fengtai",
            "shijingshan",
            "tongzhou",
            "changping",
            "daxing",
            "yizhuangkaifaqu",
            "shunyi",
            "fangshan",
            "mentougou",
            "pinggu",
            "huairou",
            "miyun",
            "yanqing",
        ],
        "sh": [
            "pudong",
            "minhang",
            "baoshan",
            "xuhui",
            "putuo",
            "yangpu",
            "changning",
            "songjiang",
            "jiading",
            "huangpu",
            "jingan",
            "hongkou",
            "qingpu",
            "fengxian",
            "jinshan",
            "chongming",
        ],
    }  # 现有的城市支持列表，只支持上海和北京
    df_community_all = pd.DataFrame()
    data_cache_dir = os.path.join("..\.cache\community_data")
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)
    data_cache_file = "community_list_{}_{}_{}_{}.csv".format(
        city,
        datetime.date.today().year,
        datetime.date.today().month,
        datetime.date.today().day,
    )
    # 缓存档期文件的位置

    for district in districts[city]:
        for page in range(1, 100):
            url = "https://m.lianjia.com/{city}/xiaoqu/{district}/pg{page}".format(
                city=city, district=district, page=page
            )
            headers = {
                "Referer": url,
                "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36",
            }
            resp = requests.get(url, headers=headers)
            resp.encoding = "utf-8"

            test_text = resp.text.split("window.__PRELOADED_STATE__ = ")[1]
            test_text = test_text.split(
                ';\n        </script>\n        <script src="//s1.ljcdn.com/m-new/client/js/vendor.a83c8be6.chunk.js"'
            )[0]
            test_json = json.loads(test_text)
            # 需要进行较长的修剪，才能够得到可以json的文本，然后就可以直接作为DataFrame

            df_community_cur = pd.DataFrame(test_json["xiaoquList"]["list"])
            if len(df_community_cur) == 0:
                break  # 如果到头了，就会自动停止
            df_community_cur = df_community_cur.dropna(subset=["id"])
            df_community_cur = df_community_cur[
                [
                    "id",
                    "name",
                    "alias",
                    "cityId",
                    "districtId",
                    "districtName",
                    "bizcircleId",
                    "bizcircleName",
                    "buildingTypes",
                    "buildingFinishYear",
                    "buildingCount",
                    "priceUnitAvg",
                    "priceUnitAvgStr",
                    "houseSellNum",
                    "houseRentNum",
                    "viewUrl",
                    "pointLat",
                    "pointLng",
                ]
            ]  # 修剪字段
            df_community_cur[[LONGITUDE, LATITUDE]] = df_community_cur.apply(
                lambda row: pd.Series(bd2wgs(row["pointLng"], row["pointLat"])), axis=1
            )
            # 分开进行GPS转换，尽量延长每次访问的时间
            sleep(1)
            if (
                len(df_community_all) > 1
                and df_community_all.iloc[-1]["id"] != df_community_cur.iloc[-1]["id"]
            ):
                df_community_all = df_community_all.append(
                    df_community_cur, ignore_index=True
                )
            elif len(df_community_all) < 1:
                df_community_all = df_community_all.append(
                    df_community_cur, ignore_index=True
                )
            else:
                break
            df_community_all.to_csv(os.path.join(data_cache_dir, data_cache_file))
        sleep(5)
    return df_community_all
