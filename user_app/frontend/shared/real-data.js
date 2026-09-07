/* ============================================================
   用户检测界面的共享视图工具、API helper 与进度事件处理
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
// 面向检测人员的精简导航。研发指标和实现细节不进入用户主流程。
function renderSidebar(activeKey) {
  const ICON_HOME = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>';
  const ICON_UPLOAD = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>';
  const ICON_HISTORY = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/></svg>';

  const items = [
    { key: 'dashboard', icon: ICON_HOME,    label: '首页',     href: 'index.html' },
    { key: 'upload',    icon: ICON_UPLOAD,  label: '开始检测', href: 'upload.html' },
    { key: 'history',   icon: ICON_HISTORY, label: '检测记录', href: 'alarms.html' },
  ];
  const html = items.map(it => `
    <a class="nav-item ${it.key === activeKey ? 'active' : ''}" href="${it.href}">
      <span class="nav-icon">${it.icon}</span>
      <span>${it.label}</span>
    </a>
  `).join('');
  return `
    <div class="sidebar">
      <div class="nav-section-title">检测中心</div>
      ${html}
      <div class="sidebar-footer">
        <div class="system-status">
          <span class="status-dot"></span>
          <span data-service-status>正在检查服务</span>
        </div>
        <div>仅分析流量特征，不解密通信内容</div>
      </div>
    </div>
  `;
}

// 顶部栏只保留产品身份与真实服务状态。
function renderTopbar() {
  const ICON_SHIELD = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>';
  return `
    <div class="topbar">
      <div class="logo">
        <div class="logo-icon">${ICON_SHIELD}</div>
        <span>加密流量安全检测</span>
      </div>
      <div class="spacer"></div>
      <div class="topbar-actions">
        <span class="service-pill"><span class="status-dot"></span> <span data-service-status>正在检查服务</span></span>
      </div>
    </div>
  `;
}

const LABEL_INFO = {
  benign: { name: '正常流量', short: '正常', level: 'safe', description: '未发现明显恶意行为' },
  adware: { name: '广告软件', short: '广告软件', level: 'medium', description: '可能包含广告追踪或非预期推广通信' },
  dns2tcp: { name: 'DNS 隧道（dns2tcp）', short: 'DNS 隧道', level: 'high', description: '疑似借助 DNS 通道传输数据' },
  dnscat2: { name: 'DNS 隧道（dnscat2）', short: 'DNS 隧道', level: 'high', description: '疑似隐蔽控制或数据外传' },
  iodine: { name: 'DNS 隧道（iodine）', short: 'DNS 隧道', level: 'high', description: '疑似通过 DNS 建立隐蔽通道' },
  ransomware: { name: '勒索软件', short: '勒索软件', level: 'critical', description: '疑似与勒索软件活动相关' },
  scareware: { name: '恐吓软件', short: '恐吓软件', level: 'medium', description: '疑似包含虚假安全警告或诱导行为' },
  smsmalware: { name: '短信恶意软件', short: '短信恶意软件', level: 'high', description: '疑似与短信恶意程序通信相关' },
};

const STATUS_INFO = {
  running: { name: '检测中', className: 'tag-warning' },
  done: { name: '已完成', className: 'tag-success' },
  failed: { name: '检测失败', className: 'tag-danger' },
  cancelled: { name: '已取消', className: 'tag-default' },
};

function labelInfo(label) {
  return LABEL_INFO[label] || { name: label || '未知', short: label || '未知', level: 'medium', description: '需要进一步核查' };
}

function statusInfo(status) {
  return STATUS_INFO[status] || { name: status || '未知', className: 'tag-default' };
}

function escapeHTML(value) {
  return String(value ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#039;');
}

function formatDateTime(value) {
  if (!value) return '—';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString('zh-CN', { hour12: false });
}

function displayFilename(value) {
  const base = String(value || '抓包检测').split(/[\\/]/).pop();
  return base.replace(/^upload_[0-9a-f]{16,}_/i, '').replace(/^[0-9a-f]{16,}_/i, '');
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
// 单任务推理结果（上传 pcap 的秒级检测产物）
const fetchTaskSummary = (taskId) =>
  fetchJSON(`/api/tasks/${encodeURIComponent(taskId)}/summary`);

const fetchTaskPredictions = ({ taskId, limit = 50, offset = 0, label = '' } = {}) =>
  fetchJSON(`/api/tasks/${encodeURIComponent(taskId)}/predictions?limit=${limit}&offset=${offset}&label=${encodeURIComponent(label)}`);

const fetchTaskFlows = ({ taskId, limit = 50, offset = 0 } = {}) =>
  fetchJSON(`/api/tasks/${encodeURIComponent(taskId)}/flows?limit=${limit}&offset=${offset}`);

const fetchTaskFlowDetail = (taskId, flowUid) =>
  fetchJSON(`/api/tasks/${encodeURIComponent(taskId)}/flows/${encodeURIComponent(flowUid)}`);

const fetchSystemStatus = () => fetchJSON('/api/status');

document.addEventListener('DOMContentLoaded', () => {
  fetchSystemStatus().then(status => {
    document.querySelectorAll('[data-service-status]').forEach(el => {
      el.textContent = status.message || (status.ready ? '检测服务已就绪' : '检测模型未就绪');
    });
  }).catch(() => {
    document.querySelectorAll('[data-service-status]').forEach(el => { el.textContent = '服务暂不可用'; });
  });
});

// 标签集合（用于下拉筛选）
const fetchLabels = () => fetchJSON('/api/labels');

// 任务历史列表
const fetchTasks = (limit = 20) => fetchJSON(`/api/tasks?limit=${limit}`);

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
