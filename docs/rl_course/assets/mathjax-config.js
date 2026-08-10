// MathJax 3 配置 — 必须在 tex-mml-svg.js 之前加载
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    tags: 'none',
    macros: {
      RR: '\\mathbb{R}',
      EE: '\\mathbb{E}',
      argmax: '\\operatorname*{arg\\,max}',
      argmin: '\\operatorname*{arg\\,min}',
      clip: '\\operatorname{clip}'
    }
  },
  svg: { fontCache: 'global', scale: 1.0 },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
};
