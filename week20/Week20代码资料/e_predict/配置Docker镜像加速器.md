# Docker 镜像加速器配置指南

## Windows Docker Desktop 配置方法

### 方法1：通过 Docker Desktop 图形界面配置

1. 打开 **Docker Desktop**
2. 点击右上角的 **设置图标（齿轮）**
3. 选择 **Docker Engine**
4. 在 JSON 配置中添加以下内容：

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

5. 点击 **Apply & Restart** 应用并重启

### 方法2：直接编辑配置文件

配置文件位置：`C:\Users\你的用户名\.docker\daemon.json`

如果文件不存在，创建它并添加以下内容：

```json
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com",
    "https://mirror.baidubce.com"
  ]
}
```

然后重启 Docker Desktop。

## 验证配置

在 PowerShell 中运行：

```powershell
docker info | Select-String -Pattern "Registry Mirrors"
```

如果看到镜像地址，说明配置成功。

## 常用国内镜像源

- 中科大镜像：`https://docker.mirrors.ustc.edu.cn`
- 网易镜像：`https://hub-mirror.c.163.com`
- 百度云镜像：`https://mirror.baidubce.com`
- 阿里云镜像：需要登录阿里云获取专属地址

## 配置完成后

重新运行构建命令：

```powershell
docker-compose build
```

