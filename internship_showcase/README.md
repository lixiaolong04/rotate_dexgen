# 实习总结展示页

这是一个纯静态、可离线运行的单页作品集，用于展示在星动纪元（RobotEra）的实习项目。页面素材已经过压缩，不依赖外部字体或前端框架。

## 本地预览

建议从项目根目录启动本地服务器，以保证不同浏览器都能正常加载视频：

```bash
python -m http.server 8000 --directory internship_showcase
```

然后访问 <http://localhost:8000>。按 `P` 可进入或退出全屏演示模式。

## 部署

`internship_showcase/` 整个目录可直接部署到 GitHub Pages、Vercel 或任意静态文件服务器，无需构建。入口文件为 `index.html`。

## 内容维护

- 页面结构与文字：`index.html`
- 视觉样式与响应式布局：`styles.css`
- 滚动动画、视频控制、灯箱与演示模式：`script.js`
- 压缩后的视频和图片：`assets/`

原始视频未被修改；网页只使用 `assets/` 中面向浏览器优化的副本。
