"""
测试API接口的脚本
"""

import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"

def test_health():
    """测试健康检查接口"""
    print("\n[测试] 健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), ensure_ascii=False, indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def test_analyze_csv_only():
    """测试仅CSV分析"""
    print("\n[测试] CSV数据分析...")
    
    # 创建示例CSV数据
    csv_data = """日期,销售额,订单数
2024-01-01,10000,50
2024-01-02,12000,60
2024-01-03,15000,75
2024-01-04,11000,55
2024-01-05,13000,65"""
    
    try:
        files = {
            'csv': ('test_data.csv', csv_data.encode('utf-8'), 'text/csv')
        }
        
        response = requests.post(f"{BASE_URL}/analyze", files=files)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n分析结果:")
            print(f"CSV分析: {result.get('csv_analysis', '')[:200]}...")
            return True
        else:
            print(f"错误: {response.text}")
            return False
    except Exception as e:
        print(f"错误: {str(e)}")
        return False

def test_analyze_with_image():
    """测试带图片的分析（需要实际的图片文件）"""
    print("\n[测试] 图片+CSV综合分析...")
    print("提示: 此测试需要提供实际的图片文件")
    print("可以通过Web界面上传文件进行测试")
    return True

def main():
    """主测试函数"""
    print("="*60)
    print("  多模态数据分析师Agent - API测试")
    print("="*60)
    
    # 测试健康检查
    if not test_health():
        print("\n[错误] 服务器未启动或无法访问")
        print("请确保服务器正在运行: python api_server.py")
        return
    
    # 测试CSV分析
    test_analyze_csv_only()
    
    # 提示Web界面测试
    print("\n" + "="*60)
    print("  测试完成")
    print("="*60)
    print("\n提示:")
    print("1. 访问 http://127.0.0.1:8000 使用Web界面")
    print("2. 上传图表图片和CSV文件进行完整测试")
    print("3. 查看API文档: http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    main()

