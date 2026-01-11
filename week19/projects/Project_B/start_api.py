"""
快速启动Project_B API服务
"""
import subprocess
import sys
import pathlib

def main():
    """启动API服务"""
    project_root = pathlib.Path(__file__).resolve().parent
    api_path = project_root / "src" / "api" / "app.py"
    
    print("=" * 50)
    print("Project_B API 服务启动")
    print("=" * 50)
    print(f"项目目录: {project_root}")
    print(f"API文件: {api_path}")
    print("\n服务将启动在: http://localhost:8001")
    print("API文档: http://localhost:8001/docs")
    print("\n按 Ctrl+C 停止服务")
    print("=" * 50)
    print()
    
    try:
        subprocess.run([sys.executable, str(api_path)], check=True)
    except KeyboardInterrupt:
        print("\n\n服务已停止")
    except Exception as e:
        print(f"\n启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

