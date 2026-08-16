/* ============================================================
   MFF-LightGBM 共享视图工具 + API helpers + SSE helper
   替代 mock-data.js（删除 MOCK 全局对象，新增真实接口访问）
   ============================================================ */

// ============ 配置 ============
// Same-origin when served by FastAPI; file:// fallback keeps direct opening usable.
const API_BASE = location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';

// 通用 JSON 请求封装（带错误抛出）
async function fetchJSON(path, opts) {
  const r = await fetch(API_BASE + path, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

// ============ 公共视图工具（从 mock-data.js 迁移，去掉 MOCK） ============
// 侧边栏 HTML 注入：5 项导航，inline Lucide SVG 图标，零外部依赖
function renderSidebar(activeKey) {
  const ICON_HOME = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>';
  const ICON_UPLOAD = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>';
  const ICON_ACTIVITY = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>';
  const ICON_ALERT = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>';
  const ICON_BELL = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>';

  const items = [
    { key: 'dashboard', icon: ICON_HOME,     label: '首页 Dashboard',         href: 'index.html' },
    { key: 'upload',    icon: ICON_UPLOAD,   label: '上传 PCAP',              href: 'upload.html' },
    { key: 'tls',       icon: ICON_ACTIVITY, label: 'TLS 行为分析',           href: 'tls-analysis.html' },
    { key: 'detection', icon: ICON_ALERT,    label: '检测结果（含模型评估）', href: 'detection.html' },
    { key: 'alarms',    icon: ICON_BELL,     label: '告警 / 任务',            href: 'alarms.html' },
  ];
  const html = items.map(it => `
    <a class="nav-item ${it.key === activeKey ? 'active' : ''}" href="${it.href}">
      <span class="nav-icon">${it.icon}</span>
      <span>${it.label}</span>
    </a>
  `).join('');
  return `
    <div class="sidebar">
      <div class="nav-section-title">主功能</div>
      ${html}
      <div class="sidebar-footer">
        <div class="system-status">
          <span class="status-dot"></span>
          <span>模型服务在线</span>
        </div>
        <div>MFF-LightGBM · 八类流量检测</div>
      </div>
    </div>
  `;
}

// 顶部栏：替换 emoji 🛡 为 inline Lucide shield SVG
function renderTopbar() {
  const ICON_SHIELD = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>';
  return `
    <div class="topbar">
      <div class="logo">
        <div class="logo-icon">${ICON_SHIELD}</div>
        <span>加密流量异常检测系统</span>
        <span class="logo-sub">MFF-LightGBM</span>
      </div>
      <div class="spacer"></div>
      <div class="topbar-actions">
        <span class="version-tag">v1.2.0</span>
        <button class="icon-btn" title="通知">🔔</button>
        <button class="icon-btn" title="帮助">?</button>
        <div class="user-chip">
          <div class="avatar">T</div>
          <span>test 用户</span>
        </div>
      </div>
    </div>
  `;
}

// 公共 ECharts 暗色主题注册（原样迁移）
function registerDarkTheme() {
  const opts = {
    backgroundColor: 'transparent',
    textStyle: { color: '#9bb0cc', fontFamily: 'Inter, sans-serif' },
    title: { textStyle: { color: '#e6edf7' }, subtextStyle: { color: '#6a82a3' } },
    legend: { textStyle: { color: '#9bb0cc' } },
    tooltip: {
      backgroundColor: 'rgba(15, 31, 58, 0.95)',
      borderColor: '#2a4068',
      textStyle: { color: '#e6edf7' }
    },
    categoryAxis: {
      axisLine: { lineStyle: { color: '#2a4068' } },
      axisTick: { lineStyle: { color: '#2a4068' } },
      axisLabel: { color: '#9bb0cc' },
      splitLine: { show: false }
    },
    valueAxis: {
      axisLine: { lineStyle: { color: '#2a4068' } },
      axisLabel: { color: '#9bb0cc' },
      splitLine: { lineStyle: { color: '#1e3050', type: 'dashed' } }
    },
    color: ['#00d4ff', '#00e676', '#ffab40', '#ff5252', '#b388ff', '#5e9eff']
  };
  if (window.echarts) echarts.registerTheme('soc-dark', opts);
}

// 风险分颜色（原样迁移）
function riskColor(score) {
  if (score >= 85) return '#ff5252';
  if (score >= 60) return '#ffab40';
  if (score >= 30) return '#ffd54f';
  return '#00e676';
}

// 风险分标签（原样迁移）
function riskLabel(score) {
  if (score >= 85) return '高危';
  if (score >= 60) return '可疑';
  if (score >= 30) return '低风险';
  return '正常';
}

// ============ API helpers ============
// 元数据列表
const fetchMetadata = (limit = 50, offset = 0) =>
  fetchJSON(`/api/metadata?limit=${limit}&offset=${offset}`);

// 元数据详情（单条）
const fetchMetadataDetail = (flowUid) =>
  fetchJSON(`/api/metadata/${encodeURIComponent(flowUid)}`);

// 预测结果列表（支持 source/limit/offset/label 过滤）
const fetchPredictions = ({ source = 'runtime', limit = 50, offset = 0, label = '' } = {}) =>
  fetchJSON(`/api/predictions?source=${source}&limit=${limit}&offset=${offset}&label=${encodeURIComponent(label)}`);

// 模型评估指标
const fetchEvaluation = (source = 'static') =>
  fetchJSON(`/api/evaluation?source=${source}`);

// 首页仪表盘聚合数据
const fetchDashboard = () => fetchJSON('/api/dashboard');

// 标签集合（用于下拉筛选）
const fetchLabels = () => fetchJSON('/api/labels');

// 任务历史列表
const fetchTasks = (limit = 20) => fetchJSON(`/api/tasks?limit=${limit}`);

// 评估图片 URL（混淆矩阵 / ROC / PR 等，直接由 <img src> 使用）
function evalImageURL(name, source = 'static') {
  return `${API_BASE}/api/evaluation/image/${encodeURIComponent(name)}?source=${source}`;
}

// ============ SSE helper ============
// 任务进度流：监听 task_start / stage_* / task_done / task_error
// handlers: { onEvent(evtName, data), onError(err), onDone(evtName, data) }
// 返回 EventSource 实例，调用方可主动 close()
function streamTaskProgress(taskId, { onEvent, onError, onDone }) {
  const src = new EventSource(`${API_BASE}/api/tasks/${encodeURIComponent(taskId)}/stream`);
  const dispatch = (evtName) => {
    src.addEventListener(evtName, (e) => {
      let data = {};
      try { data = JSON.parse(e.data || '{}'); } catch (_) { /* keep empty */ }
      onEvent && onEvent(evtName, data);
      if (evtName === 'task_done' || evtName === 'task_error') {
        src.close();
        onDone && onDone(evtName, data);
      }
    });
  };
  ['task_start', 'stage_start', 'stage_progress', 'stage_done',
   'task_done', 'task_error'].forEach(dispatch);
  src.onerror = (e) => { onError && onError(e); src.close(); };
  return src;
}
