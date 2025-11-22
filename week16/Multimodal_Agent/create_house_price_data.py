"""
创建房价数据CSV文件
生成包含多个城市、不同房型的房价数据
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def create_house_price_data(num_records=500):
    """创建房价数据"""
    
    # 城市列表
    cities = ['北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '武汉', 
              '西安', '重庆', '天津', '苏州', '长沙', '郑州', '青岛', '大连']
    
    # 区域列表（每个城市有几个区域）
    districts = {
        '北京': ['朝阳区', '海淀区', '西城区', '东城区', '丰台区', '石景山区'],
        '上海': ['浦东新区', '黄浦区', '静安区', '徐汇区', '长宁区', '虹口区'],
        '广州': ['天河区', '越秀区', '海珠区', '荔湾区', '白云区', '番禺区'],
        '深圳': ['南山区', '福田区', '罗湖区', '宝安区', '龙岗区', '龙华区'],
        '杭州': ['西湖区', '上城区', '下城区', '江干区', '拱墅区', '余杭区'],
        '南京': ['鼓楼区', '玄武区', '建邺区', '秦淮区', '雨花台区', '栖霞区'],
        '成都': ['锦江区', '青羊区', '金牛区', '武侯区', '成华区', '高新区'],
        '武汉': ['江岸区', '江汉区', '硚口区', '汉阳区', '武昌区', '洪山区'],
        '西安': ['碑林区', '莲湖区', '新城区', '雁塔区', '未央区', '灞桥区'],
        '重庆': ['渝中区', '江北区', '南岸区', '沙坪坝区', '九龙坡区', '大渡口区'],
        '天津': ['和平区', '河西区', '南开区', '河东区', '河北区', '红桥区'],
        '苏州': ['姑苏区', '工业园区', '高新区', '相城区', '吴中区', '吴江区'],
        '长沙': ['芙蓉区', '天心区', '岳麓区', '开福区', '雨花区', '望城区'],
        '郑州': ['中原区', '二七区', '管城区', '金水区', '上街区', '惠济区'],
        '青岛': ['市南区', '市北区', '黄岛区', '崂山区', '李沧区', '城阳区'],
        '大连': ['中山区', '西岗区', '沙河口区', '甘井子区', '旅顺口区', '金州区']
    }
    
    # 房屋类型
    house_types = ['公寓', '别墅', '普通住宅', '商住两用', '写字楼']
    
    # 装修情况
    decoration_types = ['精装修', '简装修', '毛坯', '豪华装修']
    
    # 朝向
    orientations = ['南', '北', '东', '西', '东南', '西南', '东北', '西北']
    
    data = []
    
    for i in range(num_records):
        # 随机选择城市
        city = random.choice(cities)
        district = random.choice(districts[city])
        
        # 房屋类型
        house_type = random.choice(house_types)
        
        # 面积（平方米）- 根据房屋类型调整范围
        if house_type == '别墅':
            area = round(random.uniform(150, 500), 1)
        elif house_type == '公寓':
            area = round(random.uniform(30, 80), 1)
        elif house_type == '写字楼':
            area = round(random.uniform(50, 300), 1)
        else:
            area = round(random.uniform(60, 200), 1)
        
        # 房间数
        if house_type == '别墅':
            bedrooms = random.choice([3, 4, 5, 6, 7])
        elif house_type == '公寓':
            bedrooms = random.choice([1, 2])
        elif house_type == '写字楼':
            bedrooms = 0  # 写字楼没有卧室
        else:
            bedrooms = random.choice([2, 3, 4])
        
        # 客厅数
        if house_type == '别墅':
            living_rooms = random.choice([2, 3, 4])
        elif house_type == '公寓':
            living_rooms = 1
        else:
            living_rooms = random.choice([1, 2])
        
        # 卫生间数
        if house_type == '别墅':
            bathrooms = random.choice([2, 3, 4, 5])
        elif house_type == '公寓':
            bathrooms = random.choice([1, 2])
        else:
            bathrooms = random.choice([1, 2, 3])
        
        # 楼层
        floor = random.randint(1, 35)
        total_floors = random.randint(floor, 35)
        
        # 建筑年份
        build_year = random.randint(1990, 2024)
        age = 2024 - build_year
        
        # 朝向
        orientation = random.choice(orientations)
        
        # 装修情况
        decoration = random.choice(decoration_types)
        
        # 价格计算（基于多个因素）
        # 基础价格（每平方米）
        base_price_per_sqm = {
            '北京': 60000, '上海': 55000, '深圳': 50000, '广州': 35000,
            '杭州': 30000, '南京': 25000, '成都': 15000, '武汉': 12000,
            '西安': 10000, '重庆': 9000, '天津': 18000, '苏州': 20000,
            '长沙': 11000, '郑州': 10000, '青岛': 15000, '大连': 12000
        }
        
        price_per_sqm = base_price_per_sqm[city]
        
        # 根据区域调整（市中心更贵）
        if district in ['朝阳区', '海淀区', '浦东新区', '黄浦区', '天河区', '南山区', '福田区']:
            price_per_sqm *= 1.3
        elif district in ['西城区', '东城区', '静安区', '徐汇区', '越秀区', '罗湖区']:
            price_per_sqm *= 1.2
        
        # 根据房屋类型调整
        if house_type == '别墅':
            price_per_sqm *= 1.5
        elif house_type == '公寓':
            price_per_sqm *= 0.9
        elif house_type == '写字楼':
            price_per_sqm *= 0.8
        
        # 根据装修情况调整
        decoration_multiplier = {
            '豪华装修': 1.2,
            '精装修': 1.1,
            '简装修': 1.0,
            '毛坯': 0.9
        }
        price_per_sqm *= decoration_multiplier[decoration]
        
        # 根据房龄调整（新房更贵）
        if age <= 5:
            price_per_sqm *= 1.1
        elif age <= 10:
            price_per_sqm *= 1.0
        elif age <= 20:
            price_per_sqm *= 0.95
        else:
            price_per_sqm *= 0.85
        
        # 根据朝向调整（南向更贵）
        if orientation in ['南', '东南', '西南']:
            price_per_sqm *= 1.05
        
        # 根据楼层调整（中间楼层更贵）
        if 5 <= floor <= 20:
            price_per_sqm *= 1.02
        elif floor <= 3:
            price_per_sqm *= 0.98
        
        # 添加随机波动
        price_per_sqm *= random.uniform(0.9, 1.1)
        
        # 计算总价
        total_price = round(area * price_per_sqm, 0)
        
        # 单价（每平方米）
        unit_price = round(price_per_sqm, 0)
        
        # 挂牌日期（最近一年内）
        listing_date = datetime.now() - timedelta(days=random.randint(0, 365))
        
        # 是否有电梯
        has_elevator = '有' if total_floors >= 7 else random.choice(['有', '无'])
        
        # 是否有车位
        has_parking = random.choice(['有', '无'])
        
        # 是否有学区
        has_school_district = random.choice(['是', '否'])
        
        data.append({
            '城市': city,
            '区域': district,
            '房屋类型': house_type,
            '面积(平方米)': area,
            '房间数': bedrooms,
            '客厅数': living_rooms,
            '卫生间数': bathrooms,
            '楼层': floor,
            '总楼层': total_floors,
            '建筑年份': build_year,
            '房龄(年)': age,
            '朝向': orientation,
            '装修情况': decoration,
            '总价(元)': int(total_price),
            '单价(元/平方米)': int(unit_price),
            '是否有电梯': has_elevator,
            '是否有车位': has_parking,
            '是否有学区': has_school_district,
            '挂牌日期': listing_date.strftime('%Y-%m-%d')
        })
    
    df = pd.DataFrame(data)
    return df

def main():
    """主函数"""
    print("正在生成房价数据...")
    
    # 生成500条记录
    df = create_house_price_data(num_records=500)
    
    # 保存为CSV文件
    output_file = 'house_price_data.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"✓ 已创建 {output_file}")
    print(f"✓ 共生成 {len(df)} 条房价记录")
    print(f"\n数据预览：")
    print(df.head(10).to_string())
    print(f"\n数据统计：")
    print(f"- 城市数量: {df['城市'].nunique()}")
    print(f"- 房屋类型: {', '.join(df['房屋类型'].unique())}")
    print(f"- 价格范围: {df['总价(元)'].min():,.0f} - {df['总价(元)'].max():,.0f} 元")
    print(f"- 平均价格: {df['总价(元)'].mean():,.0f} 元")
    print(f"- 平均单价: {df['单价(元/平方米)'].mean():,.0f} 元/平方米")
    print(f"- 平均面积: {df['面积(平方米)'].mean():.1f} 平方米")
    
    # 按城市统计
    print(f"\n各城市平均房价：")
    city_stats = df.groupby('城市').agg({
        '总价(元)': 'mean',
        '单价(元/平方米)': 'mean',
        '面积(平方米)': 'mean'
    }).round(0)
    city_stats.columns = ['平均总价', '平均单价', '平均面积']
    city_stats = city_stats.sort_values('平均总价', ascending=False)
    print(city_stats.to_string())
    
    print(f"\n数据文件已保存到: {output_file}")
    print("可以使用此文件测试多模态数据分析系统！")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"生成数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()

