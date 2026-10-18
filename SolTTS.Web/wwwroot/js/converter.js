// ============================================================
// SolTTS WebUI — Converter Module
// Model converter and file picker
// ============================================================

const Converter = {
    fpCallback: null,
    fpPattern: null,
    fpCurrentDir: null,
    fpSelectedFile: null,

    init() {
        this.checkEnv();
        document.getElementById('conv-ckpt-browse').addEventListener('click', () => {
            this.openFilePicker('*.ckpt', (path) => document.getElementById('conv-ckpt').value = path);
        });
        document.getElementById('conv-pth-browse').addEventListener('click', () => {
            this.openFilePicker('*.pth', (path) => document.getElementById('conv-pth').value = path);
        });
        document.getElementById('conv-start-btn').addEventListener('click', () => this.start());

        document.getElementById('fp-confirm').addEventListener('click', () => {
            if (this.fpSelectedFile && this.fpCallback) this.fpCallback(this.fpSelectedFile);
            document.getElementById('file-picker-modal').style.display = 'none';
        });
        document.getElementById('file-picker-close').addEventListener('click', () => {
            document.getElementById('file-picker-modal').style.display = 'none';
        });
        document.getElementById('fp-up').addEventListener('click', () => {
             if (this.fpCurrentDir) {
                 const separator = this.fpCurrentDir.includes('\\') ? '\\' : '/';
                 const parent = this.fpCurrentDir.split(/[\/\\]/).slice(0, -1).join(separator) || separator;
                 this.browseDir(parent);
             }
        });
    },

    async checkEnv() {
        const el = document.getElementById('converter-env-status');
        if (!el) return;
        try {
            const r = await fetch('/api/tts/converter/status');
            const data = await r.json();
            if (data.available) el.innerHTML = `<span style="color:#22c55e">${t('✓ Python environment ready ({path})', { path: data.pythonPath })}</span>`;
            else el.innerHTML = `<span style="color:#ef4444">${t('⚠ Conversion environment incomplete')}<br/>Python: ${data.pythonPath}<br/>Script: ${data.scriptPath}</span>`;
        } catch (e) { el.innerHTML = `<span style="color:#f59e0b">${t('⚠ Environment check request failed: {message}', { message: e.message })}</span>`; }
    },

    openFilePicker(pattern, callback) {
        this.fpCallback = callback;
        this.fpPattern = pattern;
        this.fpSelectedFile = null;
        document.getElementById('fp-selected').textContent = '';
        document.getElementById('fp-confirm').disabled = true;
        document.getElementById('file-picker-modal').style.display = 'flex';
        document.getElementById('file-picker-title').textContent = t('Select {pattern} File', { pattern });
        this.browseDir(null);
    },

    async browseDir(path) {
        const list = document.getElementById('fp-list');
        const driveList = document.getElementById('fp-drives');
        list.innerHTML = `<div class="loading">${t('Loading...')}</div>`;
        
        try {
            const queryObj = {};
            if (path) queryObj.path = path;
            if (this.fpPattern) queryObj.pattern = this.fpPattern;
            const query = new URLSearchParams(queryObj).toString();
            
            const r = await fetch(`/api/tts/fs/browse${query ? '?' + query : ''}`);
            const data = await r.json();

            this.fpCurrentDir = data.current;
            document.getElementById('fp-current-path').textContent = data.current;

            // Drives
            if (data.drives) {
                driveList.innerHTML = data.drives.map(d => `<button class="text-btn drive-btn ${data.current.startsWith(d)?'active':''}" onclick="Converter.browseDir('${d.replace(/\\/g,'\\\\')}')">${d}</button>`).join('');
            }

            // Folders and Files
            list.innerHTML = '';
            (data.entries || []).forEach(e => {
                const item = document.createElement('div');
                item.className = 'fp-item ' + (e.isDir ? 'dir' : 'file');
                
                // The backend does not provide fullPath; build it on the client
                const separator = data.current.includes('\\') ? '\\' : '/';
                const fullPath = data.current.replace(/[\/\\]$/, '') + separator + e.name;

                item.innerHTML = `
                    <span class="icon">${e.isDir ? '📁' : '📄'}</span>
                    <span class="name">${e.name}</span>
                    <span class="size">${e.isDir ? '' : (e.size / 1024 / 1024).toFixed(1) + ' MB'}</span>
                `;
                item.onclick = () => {
                    if (e.isDir) this.browseDir(fullPath);
                    else {
                        document.querySelectorAll('.fp-item').forEach(i => i.classList.remove('selected'));
                        item.classList.add('selected');
                        this.fpSelectedFile = fullPath;
                        document.getElementById('fp-selected').textContent = e.name;
                        document.getElementById('fp-confirm').disabled = false;
                    }
                };
                list.appendChild(item);
            });
        } catch (err) { console.error(err); list.innerHTML = `<div class="error">${t('Failed to load')}</div>`; }
    },

    async start() {
        const ckpt = document.getElementById('conv-ckpt').value;
        const pth = document.getElementById('conv-pth').value;
        const avatarId = document.getElementById('conv-avatar-id').value.trim();
        if (!ckpt || !pth || !avatarId) return App.showToast(t('Select both files and a voice ID'), 'error');

        document.getElementById('converter-log').textContent = '';
        document.getElementById('converter-timer').style.display = 'block';
        document.getElementById('converter-post-actions').style.display = 'none';
        const logEl = document.getElementById('converter-log');
        const startTime = Date.now();
        const timerTask = setInterval(() => {
            const s = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('conv-time-val').textContent = `${Math.floor(s/60).toString().padStart(2,'0')}:${(s%60).toString().padStart(2,'0')}`;
        }, 1000);

        try {
            const query = new URLSearchParams({
                ckpt: ckpt,
                pth: pth,
                avatarId: avatarId,
                simplify: document.getElementById('conv-simplify').checked,
                quantize: document.getElementById('conv-quantize').checked,
                clean: document.getElementById('conv-clean').checked
            });

            const eventSource = new EventSource('/api/tts/converter/run?' + query.toString());
            eventSource.onmessage = (e) => {
                try {
                    const msg = JSON.parse(e.data);
                    const { type, data } = msg;

                    if (type === 'done') {
                        eventSource.close();
                        clearInterval(timerTask);
                        document.getElementById('converter-post-actions').style.display = 'flex';
                        document.getElementById('import-hint').style.display = 'block';
                        return;
                    }

                    if (type === 'log') {
                        logEl.innerHTML += `<div>${data}</div>`;
                    } else if (type === 'warn') {
                        logEl.innerHTML += `<div class="log-warn">${data}</div>`;
                    } else if (type === 'error') {
                        logEl.innerHTML += `<div class="log-err">${data}</div>`;
                        App.showToast(t('Conversion failed'), 'error');
                    } else if (type === 'success') {
                        App.showToast(t('Model converted successfully'));
                    } else if (type === 'refs_dir') {
                        // The backend returns an absolute path; keep it so the folder can be opened later
                        App.state.lastConverterRefsDir = data;
                    }

                    logEl.scrollTop = logEl.scrollHeight;
                } catch (err) {
                    console.error('SSE Error:', err, e.data);
                }
            };
            eventSource.onerror = (err) => { 
                console.error('SSE Connection Error:', err);
                eventSource.close(); 
                clearInterval(timerTask); 
            };
        } catch (err) { App.showToast(t('Failed to start conversion: {message}', { message: err.message }), 'error'); clearInterval(timerTask); }
    }
};

// Global exports for inline onclick
window.openConverterOutput = () => {
    if (!App.state.lastConverterRefsDir) return App.showToast(t('Conversion directory not found'), 'error');
    fetch('/api/tts/fs/open-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: App.state.lastConverterRefsDir })
    }).then(() => App.showToast(t('Opened directory: {path}', { path: App.state.lastConverterRefsDir })));
};

window.importReferenceAudio = async () => {
    const avatarId = document.getElementById('conv-avatar-id').value.trim();
    if (!avatarId) return App.showToast(t('Enter a voice ID'), 'error');
    
    // Using simple prompt/alert if showFilePicker is not refactored yet, 
    // but here we keep using the converter's own logic or App's global one.
    Converter.openFilePicker('*.wav', async (filePath) => {
        try {
            App.showLoading(t('Importing audio...'));
            const targetDir = `resources/avatars/${avatarId}/references`;
            const resp = await fetch(`/api/tts/fs/copy-file?sourcePath=${encodeURIComponent(filePath)}&targetDir=${encodeURIComponent(targetDir)}`, { method: 'POST' });
            if (!resp.ok) throw new Error(t('Import failed'));
            App.hideLoading();
            App.showToast(t('Imported successfully. Refresh the page to see the new reference audio in the voice library.'));
        } catch (err) { App.hideLoading(); App.showToast(t('Import failed: {message}', { message: err.message }), 'error'); }
    });
};
