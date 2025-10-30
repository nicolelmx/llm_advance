#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from intelligent_qa_system import IntelligentQASystem, DataGenerator

def simple_demo():
    """简单演示"""
    print(" 智能问答系统演示")
    print("=" * 50)
    
    # 创建系统
    qa_system = IntelligentQASystem()
    
    # 加载预定义FAQ
    faq_data = [
        {"id": 1, "question": "订单什么时候能到？", "answer": "订单一般3-5个工作日送达，具体时间请查看物流信息。"},
        {"id": 2, "question": "支持退换货吗？", "answer": "支持7天无理由退换货，商品需保持原包装完好。"},
        {"id": 3, "question": "如何查询订单状态？", "answer": "您可以在订单页面输入订单号查询，或联系客服帮您查询。"},
        {"id": 4, "question": "有什么优惠活动？", "answer": "目前有新用户注册送券活动，关注我们获取最新优惠信息。"},
        {"id": 5, "question": "配送费怎么算？", "answer": "订单满99元免配送费，不满99元收取8元配送费。"}
    ]
    
    qa_system.load_faq(faq_data)
    
    # 测试问题
    test_questions = [
        "我的订单什么时候到？",
        "可以退货吗？",
        "怎么查看我的订单？",
        "现在有优惠吗？",
        "运费多少钱？"
    ]
    
    print("\n📋 测试问题:")
    for i, question in enumerate(test_questions, 1):
        print(f"{i}. {question}")
        result = qa_system.process_query(question)
        
        print(f"   💡 答案: {result.answer}")
        if result.similar_questions:
            best_match = result.similar_questions[0]
            print(f"   🔍 最佳匹配: {best_match[0]} (相似度: {best_match[1]:.2f})")
        print(f"   📊 置信度: {result.confidence:.2f}")
        print()

def interactive_demo():
    """交互式演示"""
    print(" 智能问答系统 - 交互模式")
    print("=" * 50)
    print("输入 'quit' 或 'exit' 退出")
    print()
    
    # 初始化系统（使用完整数据）
    data_generator = DataGenerator()
    train_data = data_generator.generate_training_data(20)
    faq_data = data_generator.get_faq_data()
    
    qa_system = IntelligentQASystem()
    qa_system.load_faq(faq_data)
    qa_system.train_models(train_data)
    
    while True:
        try:
            question = input("👤 请输入您的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            
            if not question:
                continue
            
            result = qa_system.process_query(question)
            
            print(f"🤖 {result.answer}")
            
            if result.entities:
                entities_str = ", ".join([f"{e.text}({e.label})" for e in result.entities])
                print(f"🏷️  识别实体: {entities_str}")
            
            if result.similar_questions:
                print(f"🔍 相关问题: {result.similar_questions[0][0]}")
            
            print(f"📊 置信度: {result.confidence:.2f}")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 出现错误: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="智能问答系统演示")
    parser.add_argument("--mode", choices=["simple", "interactive"], default="simple",
                       help="演示模式: simple(简单演示) 或 interactive(交互模式)")
    
    args = parser.parse_args()
    
    if args.mode == "simple":
        simple_demo()
    else:
        interactive_demo()
