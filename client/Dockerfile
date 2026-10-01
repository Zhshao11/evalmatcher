# =============================================================================
# EvalMatcher 客户端镜像：纯静态站点 + nginx 反代 /api
#   - 没有任何构建步骤（index.html 是零依赖单页，canvas 手绘图表）
#   - API 目标由环境变量 API_UPSTREAM 决定，镜像里不写死任何地址
# =============================================================================
FROM nginx:1.27-alpine

# nginx 官方镜像会把 /etc/nginx/templates/*.template 渲染到 /etc/nginx/conf.d/。
# 删掉默认的 default.conf，避免两个 server 都抢 80 端口。
RUN rm -f /etc/nginx/conf.d/default.conf

COPY client/nginx.conf.template /etc/nginx/templates/nginx.conf.template
COPY client/index.html          /usr/share/nginx/html/index.html

# 可选：注入外部 API 地址（前后端分开部署时用）
COPY client/40-inject-api-base.sh /docker-entrypoint.d/40-inject-api-base.sh
RUN chmod +x /docker-entrypoint.d/40-inject-api-base.sh

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
