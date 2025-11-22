"""
停止API服务器的脚本
"""

import os
import sys
import socket
import subprocess

# 设置控制台编码为UTF-8（Windows兼容）
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def find_process_on_port(port):
    """查找占用指定端口的进程"""
    try:
        # Windows系统使用netstat
        if sys.platform == 'win32':
            result = subprocess.run(
                ['netstat', '-ano'],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) > 0:
                        # 提取PID（最后一列）
                        for part in reversed(parts):
                            if part.isdigit():
                                return int(part)
        else:
            # Linux/Mac系统使用lsof
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                return int(result.stdout.strip())
    except Exception as e:
        print(f"查找进程时出错: {str(e)}")
    return None

def stop_server(port=8000):
    """停止指定端口的服务器"""
    print(f"正在查找占用端口 {port} 的进程...")
    
    pid = find_process_on_port(port)
    
    if pid:
        print(f"找到进程 ID: {pid}")
        try:
            if sys.platform == 'win32':
                # Windows系统
                subprocess.run(['taskkill', '/PID', str(pid), '/F'], check=True)
                print(f"[成功] 已成功关闭进程 {pid}")
            else:
                # Linux/Mac系统
                subprocess.run(['kill', '-9', str(pid)], check=True)
                print(f"[成功] 已成功关闭进程 {pid}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[失败] 关闭进程失败: {str(e)}")
            return False
    else:
        print(f"未找到占用端口 {port} 的进程")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='停止API服务器')
    parser.add_argument(
        '--port', 
        type=int, 
        default=8000,
        help='要停止的服务器端口（默认: 8000）'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("  停止多模态数据分析师Agent服务器")
    print("="*60)
    
    if stop_server(args.port):
        print("\n服务器已停止")
    else:
        print("\n未找到运行中的服务器")
    
    print("="*60)

if __name__ == "__main__":
    main()

