# 部署指南

本文档介绍如何将硅基流动API Demo部署到生产环境。

## 📦 部署方式

### 方式1：本地部署

#### 1. 环境准备

```bash
# 安装Python 3.9+
python --version

# 安装依赖
pip install -r requirements.txt
```

#### 2. 配置环境变量

创建 `.env` 文件：
```bash
SILICONFLOW_API_KEY=your_api_key_here
TEXT_MODEL=Qwen/Qwen2.5-7B-Instruct
```

#### 3. 启动应用

```bash
# 前台运行
python demo_text.py

# 后台运行 (Linux/Mac)
nohup python demo_text.py > app.log 2>&1 &

# 后台运行 (Windows)
start /B python demo_text.py
```

#### 4. 访问应用

浏览器访问: http://localhost:7860

---

## 🐳 方式2：Docker部署

### 使用Docker

#### 1. 构建镜像

```bash
docker build -t siliconflow-demo:latest .
```

#### 2. 运行容器

```bash
docker run -d \
  --name siliconflow-demo \
  -p 7860:7860 \
  -e SILICONFLOW_API_KEY=your_api_key \
  --restart unless-stopped \
  siliconflow-demo:latest
```

#### 3. 查看日志

```bash
docker logs -f siliconflow-demo
```

#### 4. 停止/启动

```bash
# 停止
docker stop siliconflow-demo

# 启动
docker start siliconflow-demo

# 重启
docker restart siliconflow-demo
```

### 使用Docker Compose（推荐）

#### 1. 配置环境变量

创建 `.env` 文件：
```bash
SILICONFLOW_API_KEY=your_api_key_here
TEXT_MODEL=Qwen/Qwen2.5-7B-Instruct
MAX_TOKENS=4096
TEMPERATURE=0.7
```

#### 2. 启动服务

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

#### 3. 更新部署

```bash
# 重新构建
docker-compose build

# 重启服务
docker-compose up -d
```

---

## ☁️ 方式3：阿里云ECS部署

### 前置要求

- 阿里云账号
- 已创建ECS实例（建议配置：2核4G）
- 已开放7860端口的安全组规则

### 部署步骤

#### 1. 连接到ECS

```bash
ssh root@your_ecs_ip
```

#### 2. 安装Docker

```bash
# 安装Docker
curl -fsSL https://get.docker.com | bash

# 启动Docker服务
systemctl start docker
systemctl enable docker

# 安装Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

#### 3. 上传项目代码

方式A: 使用Git
```bash
git clone your_repo_url
cd siliconDemo
```

方式B: 使用SCP
```bash
# 在本地执行
scp -r siliconDemo root@your_ecs_ip:/root/
```

#### 4. 配置环境变量

```bash
cd siliconDemo
cat > .env << EOF
SILICONFLOW_API_KEY=your_api_key_here
TEXT_MODEL=Qwen/Qwen2.5-7B-Instruct
EOF
```

#### 5. 启动服务

```bash
docker-compose up -d
```

#### 6. 配置Nginx反向代理（可选）

安装Nginx:
```bash
apt install nginx  # Ubuntu/Debian
yum install nginx  # CentOS
```

配置Nginx (`/etc/nginx/sites-available/siliconflow`):
```nginx
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用配置:
```bash
ln -s /etc/nginx/sites-available/siliconflow /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx
```

#### 7. 配置SSL证书（可选）

使用Let's Encrypt免费证书:
```bash
apt install certbot python3-certbot-nginx
certbot --nginx -d your_domain.com
```

---

## ☁️ 方式4：阿里云容器服务（ACK）部署

### 前置要求

- 阿里云容器服务集群
- 容器镜像服务仓库

### 部署步骤

#### 1. 推送镜像到阿里云镜像仓库

```bash
# 登录阿里云容器镜像服务
docker login --username=your_username registry.cn-hangzhou.aliyuncs.com

# 标记镜像
docker tag siliconflow-demo:latest registry.cn-hangzhou.aliyuncs.com/your_namespace/siliconflow-demo:latest

# 推送镜像
docker push registry.cn-hangzhou.aliyuncs.com/your_namespace/siliconflow-demo:latest
```

#### 2. 创建Kubernetes配置文件

`k8s-deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: siliconflow-demo
  labels:
    app: siliconflow-demo
spec:
  replicas: 2
  selector:
    matchLabels:
      app: siliconflow-demo
  template:
    metadata:
      labels:
        app: siliconflow-demo
    spec:
      containers:
      - name: siliconflow-demo
        image: registry.cn-hangzhou.aliyuncs.com/your_namespace/siliconflow-demo:latest
        ports:
        - containerPort: 7860
        env:
        - name: SILICONFLOW_API_KEY
          valueFrom:
            secretKeyRef:
              name: siliconflow-secret
              key: api-key
        resources:
          limits:
            cpu: "2"
            memory: "2Gi"
          requests:
            cpu: "1"
            memory: "1Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: siliconflow-demo-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 7860
  selector:
    app: siliconflow-demo
```

#### 3. 创建Secret

```bash
kubectl create secret generic siliconflow-secret \
  --from-literal=api-key=your_api_key_here
```

#### 4. 部署应用

```bash
kubectl apply -f k8s-deployment.yaml
```

#### 5. 查看服务状态

```bash
# 查看Pod状态
kubectl get pods

# 查看服务
kubectl get svc

# 查看日志
kubectl logs -f deployment/siliconflow-demo
```

---

## 🔒 安全配置

### 1. 环境变量保护

**不要**将 `.env` 文件提交到Git仓库：
```bash
# .gitignore
.env
.env.local
```

### 2. API密钥轮换

定期更换API密钥：
1. 在硅基流动控制台创建新密钥
2. 更新 `.env` 文件
3. 重启应用

### 3. 访问控制

#### 方式A: Nginx基础认证

```nginx
location / {
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:7860;
}
```

创建密码文件:
```bash
apt install apache2-utils
htpasswd -c /etc/nginx/.htpasswd username
```

#### 方式B: IP白名单

```nginx
location / {
    allow 1.2.3.4;  # 允许的IP
    deny all;        # 拒绝其他
    proxy_pass http://localhost:7860;
}
```

---

## 📊 监控和日志

### 1. 应用日志

#### Docker日志

```bash
# 查看实时日志
docker logs -f siliconflow-demo

# 查看最近100行
docker logs --tail 100 siliconflow-demo

# 导出日志
docker logs siliconflow-demo > app.log
```

#### 持久化日志

修改 `docker-compose.yml`:
```yaml
volumes:
  - ./logs:/app/logs
```

### 2. 系统监控

#### 使用Prometheus + Grafana

1. 安装Prometheus
2. 配置应用指标导出
3. 使用Grafana可视化

#### 简单监控脚本

```bash
#!/bin/bash
# monitor.sh

while true; do
    if ! docker ps | grep -q siliconflow-demo; then
        echo "Container is down, restarting..."
        docker-compose up -d
    fi
    sleep 60
done
```

---

## 🔄 持续部署

### GitHub Actions示例

`.github/workflows/deploy.yml`:
```yaml
name: Deploy to Aliyun

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Deploy to server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ${{ secrets.SERVER_USER }}
        key: ${{ secrets.SSH_PRIVATE_KEY }}
        script: |
          cd /root/siliconDemo
          git pull
          docker-compose down
          docker-compose build
          docker-compose up -d
```

---

## 🐛 故障排查

### 容器无法启动

```bash
# 查看容器日志
docker logs siliconflow-demo

# 检查容器状态
docker inspect siliconflow-demo

# 进入容器调试
docker exec -it siliconflow-demo /bin/bash
```

### 端口被占用

```bash
# 查看端口占用
netstat -tulpn | grep 7860

# 或使用lsof
lsof -i :7860

# 杀死占用进程
kill -9 <PID>
```

### 内存不足

```bash
# 查看容器资源使用
docker stats siliconflow-demo

# 增加资源限制
docker-compose.yml中调整resources配置
```

---

## 📈 性能优化

### 1. 使用更小的模型

```bash
TEXT_MODEL=Qwen/Qwen2.5-7B-Instruct  # 更快
```

### 2. 启用缓存

在应用中添加响应缓存机制

### 3. 负载均衡

部署多个实例，使用Nginx进行负载均衡

### 4. CDN加速

静态资源使用阿里云CDN加速

---

## 💰 成本优化

### 1. 选择合适的实例规格

- 开发环境: 1核2G
- 生产环境: 2核4G
- 高负载: 4核8G

### 2. 使用按量付费

初期使用按量付费，稳定后转包年包月

### 3. 监控API调用量

定期检查token使用量，避免超额

---

## 📞 技术支持

- 阿里云文档: https://help.aliyun.com
- Docker文档: https://docs.docker.com
- 硅基流动文档: https://docs.siliconflow.cn

---

**部署成功后记得测试所有功能！🎉**

