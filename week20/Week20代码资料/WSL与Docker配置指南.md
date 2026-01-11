# WSL 与 Docker 配置指南

## 一、WSL 安装和配置

### 1.1 当前状态

根据您的终端输出，WSL 已经安装，但还没有安装具体的 Linux 发行版。

### 1.2 选择并安装 Linux 发行版

**推荐选择**：
- **Ubuntu 22.04 LTS**：长期支持版本，稳定可靠，适合生产环境
- **Ubuntu 24.04 LTS**：最新 LTS 版本，功能更新
- **Ubuntu**：最新稳定版本（通常是 24.04）

**安装命令**：

```powershell
# 安装 Ubuntu 22.04 LTS（推荐）
wsl --install -d Ubuntu-22.04

# 或安装最新版 Ubuntu
wsl --install -d Ubuntu

# 或安装 Ubuntu 24.04 LTS
wsl --install -d Ubuntu-24.04
```

**安装过程**：
1. 执行命令后，会下载发行版（约 200-500MB）
2. 下载完成后，会自动启动并提示设置用户名和密码
3. 设置完成后，WSL 环境就准备好了

### 1.3 验证 WSL 安装

```powershell
# 查看已安装的发行版
wsl --list --verbose

# 输出示例：
#   NAME            STATE           VERSION
# * Ubuntu-22.04    Running         2

# 进入 WSL
wsl

# 或指定发行版
wsl -d Ubuntu-22.04

# 在 WSL 中检查 Linux 版本
wsl uname -a
```

### 1.4 设置默认发行版和 WSL 版本

```powershell
# 设置默认 WSL 版本为 2（推荐，性能更好）
wsl --set-default-version 2

# 设置默认发行版
wsl --set-default Ubuntu-22.04

# 查看 WSL 版本信息
wsl --status
```

---

## 二、Docker 与 WSL 的集成

### 2.1 Docker Desktop 与 WSL 2

**Docker Desktop for Windows** 支持两种后端：
1. **WSL 2 后端**（推荐）：在 WSL 2 中运行 Docker，性能更好
2. **Hyper-V 后端**：传统方式，需要 Hyper-V

**优势**：
- ✅ 更好的性能
- ✅ 更低的资源占用
- ✅ 更好的文件系统性能
- ✅ 原生 Linux 环境

### 2.2 配置 Docker Desktop 使用 WSL 2

**步骤**：

1. **安装 Docker Desktop**
   - 下载并安装 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)

2. **启动 Docker Desktop**
   - 首次启动时，会检测 WSL 2
   - 如果检测到 WSL 2，会自动使用 WSL 2 后端

3. **手动配置（如果需要）**
   - 打开 Docker Desktop
   - 进入 Settings → General
   - 勾选 "Use the WSL 2 based engine"
   - 进入 Settings → Resources → WSL Integration
   - 启用 "Enable integration with my default WSL distro"
   - 选择要集成的发行版（如 Ubuntu-22.04）
   - 点击 "Apply & Restart"

4. **验证集成**
   ```powershell
   # 在 PowerShell 中
   docker --version
   docker ps

   # 在 WSL 中
   wsl
   docker --version
   docker ps
   ```

### 2.3 在 WSL 中直接安装 Docker Engine（不使用 Docker Desktop）

如果您不想使用 Docker Desktop，可以在 WSL 中直接安装 Docker Engine：

```bash
# 1. 进入 WSL
wsl

# 2. 更新包列表
sudo apt update

# 3. 安装必要的依赖
sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# 4. 添加 Docker 官方 GPG 密钥
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# 5. 添加 Docker 仓库
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 6. 安装 Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 7. 启动 Docker 服务
sudo service docker start

# 8. 将当前用户添加到 docker 组（避免每次使用 sudo）
sudo usermod -aG docker $USER

# 9. 验证安装
docker --version
docker ps

# 10. 测试运行一个容器
docker run hello-world
```

**注意**：使用这种方式需要每次启动 WSL 时手动启动 Docker 服务，或者配置自动启动。

---

## 三、在 WSL 中使用 Docker 运行项目

### 3.1 访问 Windows 文件系统

WSL 可以访问 Windows 文件系统，路径映射如下：

```
Windows 路径: E:\培训\week20\ProjectA
WSL 路径:     /mnt/e/培训/week20/ProjectA
```

**注意**：路径中的中文字符在 WSL 中可能需要特殊处理。

### 3.2 在 WSL 中运行 Docker 命令

```bash
# 1. 进入 WSL
wsl

# 2. 切换到项目目录
cd /mnt/e/培训/week20/ProjectA

# 3. 构建 Docker 镜像
docker build -t risk-api:latest .

# 4. 运行容器
docker run -d -p 8000:8000 --name risk-api-container risk-api:latest

# 5. 查看日志
docker logs risk-api-container

# 6. 测试 API
curl http://localhost:8000/health
```

### 3.3 使用 Docker Compose

```bash
# 在 WSL 中
cd /mnt/e/培训/week20/ProjectA
docker-compose up -d
docker-compose logs -f
```

### 3.4 文件路径注意事项

**问题**：Windows 路径中的中文字符可能导致问题

**解决方法**：

1. **使用相对路径**（推荐）
   ```bash
   # 在 WSL 中，先 cd 到项目目录
   cd /mnt/e/培训/week20/ProjectA
   docker build -t risk-api:latest .
   ```

2. **使用引号包裹路径**
   ```bash
   docker build -t risk-api:latest "/mnt/e/培训/week20/ProjectA"
   ```

3. **创建符号链接**（避免中文路径）
   ```bash
   # 创建符号链接
   ln -s /mnt/e/培训/week20 /home/$USER/projects
   cd ~/projects/ProjectA
   docker build -t risk-api:latest .
   ```

---

## 四、推荐配置方案

### 方案1：Docker Desktop + WSL 2（推荐）

**优点**：
- ✅ 图形界面，易于管理
- ✅ 自动集成 WSL 2
- ✅ 自动启动 Docker 服务
- ✅ 支持 Docker Compose
- ✅ 资源管理方便

**步骤**：
1. 安装 WSL 2 和 Ubuntu 22.04
2. 安装 Docker Desktop
3. 配置 Docker Desktop 使用 WSL 2 后端
4. 在 WSL 或 PowerShell 中使用 Docker 命令

### 方案2：WSL 2 + Docker Engine（轻量级）

**优点**：
- ✅ 不需要 Docker Desktop
- ✅ 资源占用更少
- ✅ 完全命令行操作

**缺点**：
- ❌ 需要手动启动 Docker 服务
- ❌ 没有图形界面

**步骤**：
1. 安装 WSL 2 和 Ubuntu 22.04
2. 在 WSL 中安装 Docker Engine
3. 配置 Docker 服务自动启动
4. 在 WSL 中使用 Docker 命令

---

## 五、BIOS/UEFI 虚拟化设置（重要）

### 5.0 启用虚拟化功能

**为什么需要启用虚拟化？**
- ✅ WSL 2 需要虚拟化支持才能运行
- ✅ Docker Desktop 需要虚拟化支持
- ✅ VMware、VirtualBox 等虚拟机软件需要虚拟化支持
- ✅ 提高虚拟化性能

**错误提示**：
- VMware：`此主机支持 AMD-V, 但 AMD-V 处于禁用状态`
- WSL：`WSL 2 installation is incomplete`
- Docker：`Hardware assisted virtualization and data execution protection must be enabled in the BIOS`

---

### 5.0.1 检查虚拟化是否已启用

**方法1：使用任务管理器**
1. 按 `Ctrl + Shift + Esc` 打开任务管理器
2. 切换到"性能"标签
3. 选择"CPU"
4. 查看右下角的"虚拟化"状态
   - ✅ **已启用**：可以继续使用 WSL 2 和 Docker
   - ❌ **已禁用**：需要进入 BIOS 启用

**方法2：使用 PowerShell**
```powershell
# 检查虚拟化状态
Get-ComputerInfo | Select-Object -Property "HyperV*"

# 或使用系统信息
systeminfo | findstr /C:"Hyper-V"
```

**方法3：使用命令提示符**
```cmd
systeminfo | findstr /C:"Hyper-V"
```

如果显示 `Hyper-V 要求: 检测到虚拟机监控程序。将不显示 Hyper-V 所需的功能。`，说明虚拟化已启用。

---

### 5.0.2 进入 BIOS/UEFI 设置

**不同品牌电脑的进入方法**：

| 品牌 | 按键 | 说明 |
|------|------|------|
| **Dell** | `F2` 或 `F12` | 开机时连续按 |
| **HP** | `F10` 或 `Esc` | 开机时连续按 |
| **Lenovo** | `F1` 或 `F2` | 开机时连续按 |
| **ASUS** | `F2` 或 `Delete` | 开机时连续按 |
| **Acer** | `F2` 或 `Delete` | 开机时连续按 |
| **MSI** | `Delete` | 开机时连续按 |
| **Gigabyte** | `Delete` | 开机时连续按 |

**通用方法**：
1. **完全关闭电脑**（不是重启）
2. **开机时立即连续按**上述按键（不要等到 Windows 启动）
3. 如果进入 Windows，说明按晚了，需要重新尝试

**Windows 10/11 高级启动方法**：
```powershell
# 以管理员身份运行 PowerShell
shutdown /r /o /t 0
```
然后选择：`疑难解答` → `高级选项` → `UEFI 固件设置`

---

### 5.0.3 在 BIOS 中启用虚拟化（AMD 处理器）

**AMD 处理器的虚拟化功能名称**：
- `AMD-V`（最常见）
- `SVM Mode`（AMD-V 的另一种名称）
- `Virtualization Technology`
- `Secure Virtual Machine`

**查找位置**（不同主板可能不同）：
1. **Advanced** → **CPU Configuration** → **AMD-V** → **Enabled**
2. **Advanced** → **SVM Mode** → **Enabled**
3. **Security** → **Virtualization** → **Enabled**
4. **Processor Configuration** → **AMD-V** → **Enabled**

**具体步骤**：
1. 进入 BIOS 设置
2. 找到 **Advanced**（高级）或 **CPU Configuration**（CPU 配置）菜单
3. 查找以下选项之一：
   - `AMD-V`
   - `SVM Mode`
   - `Virtualization Technology`
   - `Secure Virtual Machine`
4. 将状态改为 **Enabled**（启用）
5. 保存并退出（通常是 `F10`，然后选择 `Yes`）

**常见 BIOS 界面示例**：
```
Advanced
  └── CPU Configuration
      └── AMD-V [Disabled]  ← 改为 Enabled
          └── SVM Mode [Disabled]  ← 改为 Enabled（如果有）
```

---

### 5.0.4 在 BIOS 中启用虚拟化（Intel 处理器）

**Intel 处理器的虚拟化功能名称**：
- `Intel Virtualization Technology`（Intel VT-x）
- `VT-x`
- `Virtualization Technology`
- `Intel VT`

**查找位置**（不同主板可能不同）：
1. **Advanced** → **CPU Configuration** → **Intel Virtualization Technology** → **Enabled**
2. **Advanced** → **Virtualization** → **Enabled**
3. **Security** → **Virtualization** → **Enabled**
4. **Processor Configuration** → **Intel VT-x** → **Enabled**

**具体步骤**：
1. 进入 BIOS 设置
2. 找到 **Advanced**（高级）或 **CPU Configuration**（CPU 配置）菜单
3. 查找以下选项之一：
   - `Intel Virtualization Technology`
   - `VT-x`
   - `Virtualization Technology`
4. 将状态改为 **Enabled**（启用）
5. 保存并退出（通常是 `F10`，然后选择 `Yes`）

**常见 BIOS 界面示例**：
```
Advanced
  └── CPU Configuration
      └── Intel Virtualization Technology [Disabled]  ← 改为 Enabled
          └── VT-d [Disabled]  ← 也建议启用（如果有）
```

---

### 5.0.5 其他相关 BIOS 设置

**建议同时启用的功能**（如果存在）：
- ✅ **VT-d**（Intel）或 **IOMMU**（AMD）：用于设备直通
- ✅ **Secure Boot**：安全启动（某些情况下可能需要禁用才能使用 WSL 2）
- ✅ **Fast Boot**：快速启动（可以禁用以便更容易进入 BIOS）

**注意**：
- 某些主板可能需要在 **Security**（安全）菜单中启用虚拟化
- 某些主板可能需要在 **Chipset**（芯片组）菜单中启用
- 如果找不到，可以查看主板说明书或搜索"主板型号 + 虚拟化"

---

### 5.0.6 保存并重启

**保存 BIOS 设置**：
1. 按 `F10`（大多数主板）
2. 或找到 **Save & Exit**（保存并退出）菜单
3. 选择 **Yes** 确认保存
4. 电脑会自动重启

**重启后验证**：
```powershell
# 方法1：任务管理器
# Ctrl + Shift + Esc → 性能 → CPU → 查看"虚拟化"状态

# 方法2：PowerShell
systeminfo | findstr /C:"Hyper-V"

# 方法3：检查 WSL
wsl --status

# 方法4：检查虚拟化支持
Get-ComputerInfo | Select-Object -Property "HyperV*"
```

**如果仍然显示禁用**：
1. 确认 BIOS 设置已保存
2. 确认使用的是管理员账户
3. 检查是否有其他虚拟化软件冲突（如某些杀毒软件）
4. 尝试禁用 Hyper-V 后重新启用：
   ```powershell
   # 以管理员身份运行
   Disable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
   Restart-Computer
   # 重启后
   Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V -All
   Restart-Computer
   ```

---

### 5.0.7 常见问题

**Q1：找不到虚拟化选项**
- 检查主板是否支持虚拟化（较老的 CPU 可能不支持）
- 查看 CPU 型号，确认是否支持虚拟化
- 某些主板可能隐藏了该选项，需要先启用其他高级选项

**Q2：启用后仍然无法使用**
- 确认已保存 BIOS 设置并重启
- 检查 Windows 功能中是否启用了相关功能
- 尝试禁用并重新启用 Windows 虚拟化功能

**Q3：启用虚拟化后电脑变慢**
- 虚拟化本身不会影响性能
- 可能是其他原因（如 WSL 2 占用资源）
- 可以在任务管理器中查看资源使用情况

**Q4：Secure Boot 冲突**
- 某些情况下，Secure Boot 可能与 WSL 2 冲突
- 可以尝试在 BIOS 中禁用 Secure Boot（但会降低安全性）
- 或使用 WSL 1（不推荐，性能较差）

---

## 六、常见问题

### 6.1 WSL 安装错误：0x8000ffff

**错误信息**：`安装过程中出现错误。分发名称: 'Ubuntu 22.04 LTS' 错误代码: 0x8000ffff`

**原因分析**：
- Windows 更新不完整
- 虚拟化功能未启用
- WSL 组件损坏
- 系统服务异常

**解决方法**（按优先级排序）：

**方法1：启用必要的 Windows 功能**
```powershell
# 以管理员身份运行 PowerShell，执行以下命令：

# 启用虚拟化平台
Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All

# 启用 Windows 子系统
Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -All

# 重启电脑
Restart-Computer
```

**方法2：使用图形界面启用功能**
```powershell
# 打开"启用或关闭 Windows 功能"
optionalfeatures

# 或使用命令
appwiz.cpl
```
在图形界面中勾选：
- ✅ 适用于 Linux 的 Windows 子系统
- ✅ 虚拟机平台
- ✅ Hyper-V（如果可用）

**方法3：更新 WSL**
```powershell
# 以管理员身份运行
wsl --update

# 设置默认版本
wsl --set-default-version 2

# 重启后重试安装
wsl --install -d Ubuntu-22.04
```

**方法4：手动安装 WSL 组件**
```powershell
# 1. 下载 WSL2 内核更新包
# 访问：https://aka.ms/wsl2kernel
# 下载并安装 WSL2 Linux 内核更新包

# 2. 重启电脑

# 3. 重新尝试安装
wsl --install -d Ubuntu-22.04
```

**方法5：检查 BIOS/UEFI 设置**
- 重启电脑，进入 BIOS/UEFI 设置
- 启用虚拟化功能（Virtualization Technology / Intel VT-x / AMD-V）
- 保存并退出

**方法6：使用 Microsoft Store 安装（替代方案）**
```powershell
# 1. 打开 Microsoft Store
start ms-windows-store:

# 2. 搜索 "Ubuntu 22.04 LTS"
# 3. 点击"获取"或"安装"
# 4. 安装完成后，从开始菜单启动 Ubuntu
```

**方法7：清理并重新安装 WSL**
```powershell
# 以管理员身份运行

# 1. 卸载 WSL（如果已部分安装）
wsl --unregister Ubuntu-22.04

# 2. 重置 WSL
wsl --shutdown

# 3. 更新 WSL
wsl --update

# 4. 重启电脑

# 5. 重新安装
wsl --install -d Ubuntu-22.04
```

**方法8：检查 Windows 更新**
```powershell
# 检查更新
Get-WindowsUpdate

# 或使用图形界面
ms-settings:windowsupdate
```
确保 Windows 系统是最新的。

**方法9：检查系统要求**
- Windows 10 版本 2004 或更高版本（内部版本 19041 或更高）
- Windows 11（所有版本）
- 64 位系统

检查 Windows 版本：
```powershell
winver
```

**验证修复**：
```powershell
# 检查 WSL 状态
wsl --status

# 查看已安装的发行版
wsl --list --verbose

# 尝试安装
wsl --install -d Ubuntu-22.04
```

### 6.2 WSL 安装后未显示在列表中

**问题**：安装显示"已安装 Ubuntu 22.04 LTS"，但 `wsl --list --verbose` 显示没有已安装的分发版

**可能原因**：
- 安装过程未完全完成
- 需要重启电脑
- 发行版需要首次启动初始化
- WSL 服务未正常运行

**解决方法**（按优先级排序）：

**方法1：尝试直接启动 Ubuntu（推荐）**
```powershell
# 直接启动 Ubuntu，可能会触发初始化
wsl -d Ubuntu-22.04

# 或使用完整名称
wsl -d "Ubuntu 22.04 LTS"

# 或直接使用
wsl
```

如果成功启动，会提示您：
- 创建用户名
- 设置密码
- 确认密码

完成后，再次检查：
```powershell
wsl --list --verbose
```

**方法2：重启电脑**
```powershell
# 重启后再次检查
Restart-Computer

# 重启后运行
wsl --list --verbose
wsl -d Ubuntu-22.04
```

**方法3：检查并重启 WSL 服务**
```powershell
# 以管理员身份运行 PowerShell

# 检查 WSL 服务状态
Get-Service -Name LxssManager

# 如果服务未运行，启动它
Start-Service -Name LxssManager

# 或重启服务
Restart-Service -Name LxssManager

# 然后重试
wsl --list --verbose
wsl -d Ubuntu-22.04
```

**方法4：从 Microsoft Store 启动**
1. 打开 Microsoft Store（`ms-windows-store:`）
2. 搜索 "Ubuntu 22.04 LTS"
3. 如果显示"启动"而不是"获取"，点击启动
4. 这会触发首次初始化，设置用户名和密码
5. 初始化完成后，再次检查列表

**方法5：检查 WSL 状态**
```powershell
# 检查 WSL 状态
wsl --status

# 关闭所有 WSL 实例
wsl --shutdown

# 等待几秒后重试
wsl --list --verbose
wsl -d Ubuntu-22.04
```

**方法6：手动查找并注册发行版**
```powershell
# 查找 Ubuntu 安装位置（通常在 AppData）
# 检查可能的路径
$paths = @(
    "$env:LOCALAPPDATA\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_*\LocalState",
    "$env:USERPROFILE\AppData\Local\Packages\CanonicalGroupLimited.Ubuntu22.04LTS_*\LocalState"
)

# 如果找到 vhdx 文件，可以尝试手动注册
# 但通常不需要，直接启动即可
```

**方法7：清理并重新安装（最后手段）**
```powershell
# 如果以上方法都不行
wsl --unregister Ubuntu-22.04
wsl --shutdown
wsl --update
wsl --install -d Ubuntu-22.04
```

**验证修复**：
```powershell
# 检查列表
wsl --list --verbose

# 应该看到类似输出：
#   NAME            STATE           VERSION
# * Ubuntu-22.04    Running         2

# 启动 Ubuntu
wsl -d Ubuntu-22.04
```

### 6.3 WSL 2 未安装或版本不对

**错误**：`WSL 2 installation is incomplete`

**解决方法**：
```powershell
# 更新 WSL
wsl --update

# 设置默认版本为 2
wsl --set-default-version 2

# 如果已安装发行版，转换到 WSL 2
wsl --set-version Ubuntu-22.04 2
```

### 6.4 Docker Desktop 无法检测到 WSL

**解决方法**：
1. 确保 WSL 2 已安装并设置为默认版本
2. 重启 Docker Desktop
3. 在 Docker Desktop Settings → Resources → WSL Integration 中手动启用

### 6.5 文件路径问题（中文路径）

**问题**：构建镜像时找不到文件

**解决方法**：
- 使用相对路径（先 cd 到项目目录）
- 或创建符号链接避免中文路径
- 或使用引号包裹路径

### 6.6 端口访问问题

**问题**：在 WSL 中运行的容器，Windows 无法访问

**解决方法**：
- Docker Desktop 会自动处理端口映射
- 如果使用 WSL 中的 Docker Engine，确保端口映射正确
- 使用 `-p 8000:8000` 映射端口

### 6.7 性能问题

**优化建议**：
- 使用 WSL 2（而不是 WSL 1）
- 将项目文件放在 WSL 文件系统中（而不是 `/mnt/c`）
- 使用 Docker Desktop 的 WSL 2 后端

---

## 六、快速开始命令

### 安装 WSL 和 Ubuntu

```powershell
# 1. 安装 Ubuntu 22.04
wsl --install -d Ubuntu-22.04

# 2. 设置默认 WSL 版本
wsl --set-default-version 2

# 3. 验证安装
wsl --list --verbose
```

### 安装 Docker Desktop

1. 下载并安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)
2. 启动 Docker Desktop
3. 配置使用 WSL 2 后端（Settings → General）
4. 启用 WSL Integration（Settings → Resources → WSL Integration）

### 运行项目

```powershell
# 方式1：在 PowerShell 中（推荐）
cd E:\培训\week20\ProjectA
docker build -t risk-api:latest .
docker run -d -p 8000:8000 --name risk-api-container risk-api:latest

# 方式2：在 WSL 中
wsl
cd /mnt/e/培训/week20/ProjectA
docker build -t risk-api:latest .
docker run -d -p 8000:8000 --name risk-api-container risk-api:latest
```

---

## 七、总结

### 推荐配置

1. ✅ **安装 WSL 2**：`wsl --install -d Ubuntu-22.04`
2. ✅ **安装 Docker Desktop**：使用 WSL 2 后端
3. ✅ **配置集成**：在 Docker Desktop 中启用 WSL Integration
4. ✅ **使用 Docker**：在 PowerShell 或 WSL 中都可以使用 Docker 命令

### 关键命令速查

```powershell
# WSL 管理
wsl --list --verbose          # 查看已安装的发行版
wsl --install -d Ubuntu-22.04 # 安装 Ubuntu 22.04
wsl                           # 进入默认 WSL
wsl --set-default-version 2   # 设置默认 WSL 版本

# Docker（在 PowerShell 或 WSL 中都可以）
docker --version              # 检查版本
docker ps                     # 查看运行中的容器
docker build -t risk-api:latest .  # 构建镜像
docker run -d -p 8000:8000 --name risk-api-container risk-api:latest  # 运行容器
```

---

## 参考资源

- [WSL 官方文档](https://learn.microsoft.com/zh-cn/windows/wsl/)
- [Docker Desktop WSL 2 后端](https://docs.docker.com/desktop/wsl/)
- [Docker Engine 安装指南](https://docs.docker.com/engine/install/ubuntu/)

