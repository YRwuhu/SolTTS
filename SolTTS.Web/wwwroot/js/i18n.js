// ============================================================
// SolTTS WebUI — i18n Module
// Runtime language switching between English and Simplified Chinese.
// English source strings double as dictionary keys, so markup stays
// language-neutral and dynamic strings resolve through t(key, vars).
// ============================================================

const I18N = {
    lang: 'en',
    supported: ['en', 'zh'],
    storageKey: 'soltts-lang',

    // en -> zh
    dict: {
        // ---- Shell / navigation ----
        'SolTTS Control Center': 'SolTTS 控制中心',
        'Navigation': '导航',
        'Dashboard': '仪表盘',
        'Voice Library': '音色库',
        'Control Center': '控制中心',
        'Synthesis Lab': '合成实验室',
        'Model Converter': '模型转换',
        'Connecting to engine...': '正在连接引擎...',
        'Reload Config': '重载配置',
        'Toggle theme': '切换主题',
        'Close': '关闭',
        'Cancel': '取消',
        'Confirm': '确认',
        'Download': '下载',
        'Status': '状态',
        'Latency': '延迟',
        'Duration': '时长',
        'Actions': '操作',
        'ID': 'ID',
        'Name': '名称',
        'Engine': '引擎',
        'Voice': '音色',
        'Voices': '音色',
        'Control': '控制',
        'Synth': '合成',
        'Convert': '转换',
        'Browse': '浏览',
        'Select File': '选择文件',
        '↑ Up': '↑ 上级目录',
        'default': '默认',
        'Ready.': '就绪。',
        'Hz': 'Hz',
        'ms': '毫秒',
        'tokens/s': 'tokens/s',

        // ---- Dashboard ----
        'Real-time runtime status and engine resource overview': '实时运行状态与引擎资源概览',
        'Sample Rate': '采样率',
        'Current Mode': '当前模式',
        'Inference Engine': '推理引擎',
        'Registered Voices': '已注册音色',
        'Voices available': '可用音色',
        'V2 Engine': 'V2 引擎',
        'Runtime Info': '运行时信息',
        'Model paths and default voice configuration': '模型路径与默认音色配置',
        'CONFIG': '配置',
        'Default Voice (DefaultAvatarId)': '默认音色 (DefaultAvatarId)',
        'Resources Directory': '资源目录',
        'Quick Actions': '快捷操作',
        'Frequently used features': '常用功能',
        'Start Synthesis': '开始合成',
        'Open the lab and preview voices': '打开实验室并试听音色',
        'Manage Voices': '管理音色',
        'Edit voices and reference audio': '编辑音色与参考音频',
        'Convert Model': '转换模型',
        'Convert GPT-SoVITS weights to ONNX': '将 GPT-SoVITS 权重转换为 ONNX',
        'Enabled': '已启用',
        'Disabled (V1)': '未启用 (V1)',
        'Engine Online': '引擎在线',
        'Engine Offline': '引擎离线',

        // ---- Voice library ----
        'Manage voices, engine bindings, and reference audio': '管理音色、引擎绑定与参考音频',
        'Voice List': '音色列表',
        'Registered voices and reference audio counts': '已注册音色与参考音频数量',
        '+ New Voice': '+ 新建音色',
        'Reference Audio': '参考音频',
        'Configuration Details': '配置详情',
        'Voice metadata and reference audio mappings': '音色元数据与参考音频映射',
        'Save Changes': '保存修改',
        'Basic Metadata': '基本信息',
        'Voice ID (unique identifier)': '音色 ID（唯一标识）',
        'Description': '描述',
        'Engine Binding': '引擎绑定',
        'Engine Mode': '引擎模式',
        'Follow global default (GLOBAL)': '跟随全局默认 (GLOBAL)',
        'Force GPT-SoVITS (V2)': '强制 GPT-SoVITS (V2)',
        'Force Genie-TTS (V1)': '强制 Genie-TTS (V1)',
        'Default Reference Audio': '默认参考音频',
        'Reference Audio Mappings (References)': '参考音频映射 (References)',
        'Each reference includes an ID, language, and matching text': '每条参考音频包含 ID、语言与对应文本',
        '↑ Import Audio File': '↑ 导入音频文件',
        '+ Add Reference': '+ 添加参考音频',
        'Unnamed': '未命名',
        'Edit': '编辑',
        'Delete': '删除',
        'Remove': '移除',
        'Display name': '显示名称',
        'Auto / unspecified': '自动 / 未指定',
        'Chinese (zh)': '中文 (zh)',
        'English (en)': '英文 (en)',
        'Japanese (ja)': '日文 (ja)',
        '(Select file)': '（选择文件）',
        'Click to copy filename': '点击复制文件名',
        'Load failed': '加载失败',
        'Invalid voice list format': '音色列表格式无效',
        'New Voice Configuration': '新建音色配置',
        '— Select —': '— 请选择 —',
        ' ⚠ Missing model': ' ⚠ 缺少模型',
        ' ⚠ Missing voice directory': ' ⚠ 缺少音色目录',
        '{count} refs': '{count} 条参考',
        'Voice saved successfully': '音色保存成功',
        'Save failed': '保存失败',
        'Delete failed': '删除失败',
        'Saving voice and reloading configuration...': '正在保存音色并重载配置...',
        'Deleting voice...': '正在删除音色...',
        'Deleted successfully': '删除成功',
        'Select a voice ID first': '请先选择音色 ID',
        'Importing audio file...': '正在导入音频文件...',
        'File copy failed': '文件复制失败',
        'File imported into the references directory. Refreshing list...': '文件已导入 references 目录，正在刷新列表...',
        'Voice not found: {id}': '找不到音色：{id}',
        'Edit Voice: {name}': '编辑音色：{name}',
        'Copied: {text}': '已复制：{text}',
        'Save error: {message}': '保存错误：{message}',
        'Import failed: {message}': '导入失败：{message}',
        'Error: {message}': '错误：{message}',
        'Delete the configuration for voice [{id}]? This does not remove physical model files.':
            '确定删除音色 [{id}] 的配置吗？此操作不会删除物理模型文件。',

        // ---- Control center ----
        'Global engine parameters, synthesis defaults, and G2P language settings': '全局引擎参数、合成默认值与 G2P 语言设置',
        'Restore Defaults': '恢复默认',
        'Save & Apply': '保存并应用',
        'Core Runtime & Concurrency': '核心运行时与并发',
        'Resources Directory (ResourcesDir)': '资源目录 (ResourcesDir)',
        'Inference Pool Capacity (PoolCapacity)': '推理池容量 (PoolCapacity)',
        'Enable memory / session reuse': '启用内存 / 会话复用',
        'Hybrid mode: run vocoder on GPU (Vulkan/Metal)': '混合模式：声码器使用 GPU (Vulkan/Metal)',
        'BERT feature cache (reuses identical text, saves ~300ms per sentence)': 'BERT 特征缓存（复用相同文本，每句约省 300ms）',
        'Use loop-free RoBERTa model (roberta_noloop.onnx)': '使用去 Loop 的 RoBERTa 模型 (roberta_noloop.onnx)',
        'Enable emotion control (requires τ files)': '启用情感控制（需要 τ 文件）',
        'Emotion intensity α (EmotionAlpha)': '情感强度 α (EmotionAlpha)',
        'Emotion Vector Directory (EmotionVectorPath)': '情感向量目录 (EmotionVectorPath)',
        'Enable verbose debug logging (DebugMode)': '启用详细调试日志 (DebugMode)',
        'Synthesis Defaults': '合成默认值',
        'Prefer V2 engine globally': '全局优先使用 V2 引擎',
        'Default speed (Speed:': '默认语速 (Speed:',
        'Default noise scale (NoiseScale)': '默认噪声系数 (NoiseScale)',
        'Default sampling temperature (Temperature)': '默认采样温度 (Temperature)',
        'Default Top-K sampling': '默认 Top-K 采样',
        'Streaming chunk size': '流式分块大小',
        'Streaming pre-buffer chunks': '流式预缓冲块数',
        'ONNX Thread Optimization': 'ONNX 线程优化',
        'Graph optimization level (0-2)': '图优化级别 (0-2)',
        '0 - Disabled': '0 - 禁用',
        '1 - Basic': '1 - 基础',
        '2 - Aggressive': '2 - 激进',
        'Intra-op threads (IntraOp)': '算子内线程数 (IntraOp)',
        'Inter-op threads (InterOp)': '算子间线程数 (InterOp)',
        'Inference Retry Threshold Multipliers (lower is more lenient)': '推理重试阈值系数（越低越宽松）',
        'Max generation retries': '最大生成重试次数',
        'Chinese (Zh) threshold': '中文 (Zh) 阈值',
        'English (En) threshold': '英文 (En) 阈值',
        'Japanese (Ja) threshold': '日文 (Ja) 阈值',
        'Other (Other)': '其他 (Other)',
        'G2P & Language List': 'G2P 与语言列表',
        'Allowed languages (two-language limit)': '允许的语言（最多两种）',
        'Chinese': '中文',
        'English': '英文',
        'Japanese': '日文',
        'G2P priority mode': 'G2P 优先级模式',
        '0 - Dictionary first (DictFirst)': '0 - 词典优先 (DictFirst)',
        '1 - Dictionary only (DictOnly)': '1 - 仅词典 (DictOnly)',
        '2 - Model first (ModelFirst)': '2 - 模型优先 (ModelFirst)',
        'Custom dictionary path': '自定义词典路径',
        '(no voices available)': '（无可用音色）',
        'At most two languages can be selected at once': '最多只能同时选择两种语言',
        'Saving global configuration and reloading the engine...': '正在保存全局配置并重载引擎...',
        'Global configuration updated': '全局配置已更新',
        'Unknown error': '未知错误',
        'This will restore all configured settings and clear the voice library. Continue?':
            '此操作将还原所有配置项并清空音色库，是否继续？',
        'Resetting global configuration and reloading the engine...': '正在重置全局配置并重载引擎...',
        'Configuration reset to system defaults': '配置已重置为系统默认',
        'Save failed: {message}': '保存失败：{message}',
        'Update failed: {message}': '更新失败：{message}',
        'Reset failed: {message}': '重置失败：{message}',

        // ---- Synthesis lab ----
        'Enter text, preview the selected voice, and inspect inference performance': '输入文本，试听所选音色并查看推理性能',
        'Inference Options': '推理选项',
        'Target Voice': '目标音色',
        'Speed': '语速',
        'Sampling temperature (Temperature)': '采样温度 (Temperature)',
        'Noise scale (NoiseScale)': '噪声系数 (NoiseScale)',
        'Top-K samples': 'Top-K 采样数',
        'Use default': '使用默认',
        'Emotion prompt (emotion control without reference audio)': '情感提示（无需参考音频的情感控制）',
        'Neutral (no injection)': '中性（不注入）',
        'Happy': '高兴',
        'Angry': '生气',
        'Sad': '悲伤',
        'Surprised': '惊讶',
        'Fearful': '恐惧',
        'Playful': '俏皮',
        'Emotion intensity α': '情感强度 α',
        'Enable streaming playback (WIP)': '启用流式播放（开发中）',
        'Synthesis Text': '合成文本',
        'Supports mixed Chinese / English / Japanese input': '支持中 / 英 / 日混合输入',
        'Synthesize & Play': '合成并播放',
        'Processing audio stream...': '正在处理音频流...',
        'Generation Time': '生成耗时',
        'Generation Speed': '生成速度',
        'Real-Time Factor (RTF)': '实时率 (RTF)',
        'Concurrency Stress Test': '并发压力测试',
        'Concurrency (Threads)': '并发数（线程）',
        'Repeat count': '重复次数',
        'Voice mode': '音色模式',
        'Current voice': '当前音色',
        'Rotate all voices': '轮换全部音色',
        'Reference Audio ID': '参考音频 ID',
        'Test text (one per line, used round-robin; leave blank to use the text above)': '测试文本（每行一条，轮流使用；留空则使用上方文本）',
        'Run Stress Test': '运行压力测试',
        'Initializing...': '正在初始化...',
        'Running...': '执行中...',
        'Success': '成功',
        'Error': '异常',
        'Select at most two languages': '最多选择两种语言',
        'Enter text to synthesize': '请输入要合成的文本',
        'Synthesis failed': '合成失败',
        'Download synthesized audio (WAV)': '下载合成音频 (WAV)',
        'Synthesis succeeded, playing audio': '合成成功，正在播放',
        'Initializing streaming audio...': '正在初始化流式音频...',
        'Streaming request failed': '流式请求失败',
        'First-Chunk Latency': '首包延迟',
        'Streaming synthesis complete': '流式合成完成',
        'Stress test complete': '压测完成',
        'Synthesis error: {message}': '合成错误：{message}',
        'Streaming error: {message}': '流式错误：{message}',
        'Streaming ({hz}Hz): {chunks} chunks received...': '流式播放 ({hz}Hz)：已接收 {chunks} 块...',
        'Failed ({status})': '失败 ({status})',
        'Download streamed audio (WAV, {duration}s)': '下载流式音频 (WAV, {duration}s)',
        'Stress test started: concurrency {concurrency}, total {total}, {texts} text(s), {voices} voice(s)':
            '压测开始：并发 {concurrency}，总数 {total}，文本 {texts} 条，音色 {voices} 个',
        'Progress: {done}/{total}': '进度：{done}/{total}',
        'Success: {count}': '成功：{count}',
        'Failed: {count}': '失败：{count}',
        'Avg: {avg}ms | Min: {min}ms | Max: {max}ms': '平均：{avg}ms | 最小：{min}ms | 最大：{max}ms',
        'QPS: {qps} | Total: {total}s': 'QPS：{qps} | 总耗时：{total}s',

        // ---- Converter ----
        'Convert GPT-SoVITS .ckpt and .pth weights into the ONNX format required by SolTTS':
            '将 GPT-SoVITS 的 .ckpt 与 .pth 权重转换为 SolTTS 所需的 ONNX 格式',
        'Conversion Job': '转换任务',
        'Select weight files and specify the target voice ID': '选择权重文件并指定目标音色 ID',
        'GPT weight file (.ckpt)': 'GPT 权重文件 (.ckpt)',
        'SoVITS weight file (.pth)': 'SoVITS 权重文件 (.pth)',
        'Target voice ID (output directory name)': '目标音色 ID（输出目录名）',
        'Enable ONNX Simplify': '启用 ONNX 简化',
        'Enable INT8 dynamic quantization': '启用 INT8 动态量化',
        'Clean up intermediate files': '清理中间文件',
        'Start Conversion': '开始转换',
        'Open Voice Directory': '打开音色目录',
        'Import Reference Audio': '导入参考音频',
        'After importing, refresh the page to see the new reference audio in the voice library':
            '导入后刷新页面即可在音色库中看到新的参考音频',
        'Conversion Log': '转换日志',
        'Live conversion output': '实时转换输出',
        'Elapsed:': '已用时：',
        'Select {pattern} File': '选择 {pattern} 文件',
        'Loading...': '加载中...',
        'Failed to load': '加载失败',
        'Select both files and a voice ID': '请选择两个文件并填写音色 ID',
        'Conversion failed': '转换失败',
        'Model converted successfully': '模型转换成功',
        'Conversion directory not found': '未找到转换目录',
        'Enter a voice ID': '请输入音色 ID',
        'Importing audio...': '正在导入音频...',
        'Import failed': '导入失败',
        'Imported successfully. Refresh the page to see the new reference audio in the voice library.':
            '导入成功。刷新页面即可在音色库中看到新的参考音频。',
        '✓ Python environment ready ({path})': '✓ Python 环境就绪 ({path})',
        '⚠ Conversion environment incomplete': '⚠ 转换环境不完整',
        '⚠ Environment check request failed: {message}': '⚠ 环境检查请求失败：{message}',
        'Failed to start conversion: {message}': '启动转换失败：{message}',
        'Opened directory: {path}': '已打开目录：{path}',

        'Switch language': '切换语言',
        'Core Runtime &amp; Concurrency': '核心运行时与并发',
        'G2P &amp; Language List': 'G2P 与语言列表',
        'Save &amp; Apply': '保存并应用',
        'Synthesize &amp; Play': '合成并播放',
        // ---- Runtime messages ----
        'Processing, please wait...': '正在处理，请稍候...',
        'Data sync failed. Check the console for details.': '数据同步失败，请检查控制台获取详情。',
        'Reloading configuration. The engine will reinitialize...': '正在重载配置，引擎将重新初始化...',
        'Engine reloaded successfully': '引擎重载成功',
        'Click Browse to select a file...': '点击浏览选择文件...',
        'Enter text here, then click synthesize to preview the voice...': '在此输入文本，然后点击合成以试听音色...',
        'Leave blank for default': '留空使用默认值',
        'e.g. sol': '例如：sol'
    },

    reverse: {},

    init() {
        if (this._initialized) {
            // A second init (core.js and hyperos-ui.js both run on DOMContentLoaded)
            // must not bind a duplicate click handler, or one click toggles twice.
            this.apply();
            return;
        }
        this._initialized = true;

        for (const en of Object.keys(this.dict)) this.reverse[this.dict[en]] = en;

        const saved = (localStorage.getItem(this.storageKey) || '').toLowerCase();
        const browser = (navigator.language || 'en').toLowerCase();
        this.lang = this.supported.includes(saved) ? saved : (browser.startsWith('zh') ? 'zh' : 'en');
        this.apply();
        this.bindToggle();
    },

    setLang(lang, options = {}) {
        if (!this.supported.includes(lang)) return;
        if (lang === this.lang) return;
        this.lang = lang;
        localStorage.setItem(this.storageKey, lang);
        this.apply();
        // Re-render views that build text dynamically from state.
        if (typeof Dashboard !== 'undefined') Dashboard.update();
        if (typeof Avatars !== 'undefined') Avatars.renderList();
        if (typeof Config !== 'undefined') Config.renderForm();
        if (typeof Playground !== 'undefined') Playground.updateAvatars();
        if (typeof HyperOS !== 'undefined' && typeof App !== 'undefined') HyperOS.updateChrome(App.state.activeTab);
    },

    toggle() {
        this.setLang(this.lang === 'en' ? 'zh' : 'en');
    },

    translate(text) {
        if (this.lang === 'en') return this.reverse[text] || text;
        return this.dict[text] || text;
    },

    /** Translate a key that may appear HTML-entity-encoded in the source. */
    translateEncoded(text) {
        const decoded = text.replace(/&amp;/g, '&');
        if (this.lang === 'zh') {
            if (this.dict[text]) return this.dict[text];
            if (this.dict[decoded]) return this.dict[decoded];
            return text;
        }
        // English mode: return a decoded form so entities render as characters.
        if (this.reverse[text]) return this.reverse[text];
        if (this.reverse[decoded]) return this.reverse[decoded];
        return decoded;
    },

    /** Translate with {placeholder} interpolation. */
    t(text, vars) {
        let out = this.translate(text);
        if (vars) {
            for (const k of Object.keys(vars)) {
                out = out.split('{' + k + '}').join(vars[k]);
            }
        }
        return out;
    },

    apply(root = document) {
        const doc = document.documentElement;
        doc.lang = this.lang === 'zh' ? 'zh-CN' : 'en';
        doc.dataset.lang = this.lang;

        // Explicitly tagged nodes, including text adjacent to inline elements.
        for (const el of root.querySelectorAll('[data-i18n]')) {
            const key = el.getAttribute('data-i18n');
            if (!key) continue;
            const translated = this.translateEncoded(key);
            if (el.children.length === 0) {
                el.textContent = translated;
            } else {
                // Replace only the leading text node, preserving child elements.
                const first = el.firstChild;
                if (first && first.nodeType === Node.TEXT_NODE && first.nodeValue.trim()) {
                    const lead = first.nodeValue.match(/^\s*/)[0];
                    first.nodeValue = lead + translated;
                }
            }
        }

        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
        const nodes = [];
        while (walker.nextNode()) nodes.push(walker.currentNode);

        for (const node of nodes) {
            const parent = node.parentElement;
            if (!parent) continue;
            const tag = parent.tagName;
            if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'OPTGROUP') continue;
            if (parent.closest('[data-i18n-skip]')) continue;

            const raw = node.nodeValue;
            const trimmed = raw.trim();
            if (!trimmed) continue;

            // Already in the active language (re-apply is idempotent).
            if (this.lang === 'zh' && this.reverse[trimmed]) continue;

            // Skip text nodes that are part of a larger mixed-content element.
            if (parent.children.length > 0 && parent.textContent.trim() !== trimmed) continue;

            // Options carry their own display text; translate them in place.
            const translated = this.translateEncoded(trimmed);
            if (translated !== trimmed) {
                const lead = raw.match(/^\s*/)[0];
                const trail = raw.match(/\s*$/)[0];
                node.nodeValue = lead + translated + trail;
            }
        }

        // Attribute text (placeholders, tooltips, aria labels).
        const dataValues = new Set(['default', 'emotion_vectors', 'shared/custom_dict.txt', 'e.g. sol']);
        for (const el of root.querySelectorAll('[placeholder], [title], [aria-label]')) {
            if (el.closest('[data-i18n-skip]')) continue;
            for (const attr of ['placeholder', 'title', 'aria-label']) {
                const val = el.getAttribute(attr);
                if (!val || dataValues.has(val)) continue;
                const translated = this.translate(val);
                if (translated !== val) el.setAttribute(attr, translated);
            }
        }

        const label = document.getElementById('lang-toggle-label');
        if (label) label.textContent = this.lang === 'en' ? '中文' : 'EN';
        const toggle = document.getElementById('lang-toggle');
        if (toggle) toggle.title = this.lang === 'en' ? 'Switch to Chinese' : 'Switch to English';

        document.title = this.lang === 'zh' ? 'SolTTS 控制中心' : 'SolTTS Control Center';
    },

    bindToggle() {
        const btn = document.getElementById('lang-toggle');
        if (!btn) return;
        btn.addEventListener('click', () => this.toggle());
    }
};

/** Shorthand for dynamic strings. */
function t(text, vars) { return I18N.t(text, vars); }

if (typeof window !== 'undefined') {
    window.I18N = I18N;
    window.t = t;
}
