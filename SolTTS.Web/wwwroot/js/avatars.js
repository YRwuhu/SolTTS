// ============================================================
// SolTTS WebUI — Avatars Module
// Voice library management, editor, and reference audio field completion (Name/Language)
// ============================================================

const Avatars = {
    init() {
        document.getElementById('add-avatar-btn').addEventListener('click', () => this.openEditor());
        document.getElementById('cancel-avatar-btn').addEventListener('click', () => {
            document.getElementById('avatar-editor').style.display = 'none';
        });

        document.getElementById('add-reference-btn').addEventListener('click', () => {
            this.syncFromDOM();
            App.state.currentEditingAvatar.references.push({
                id: 'ref_' + Date.now(),
                name: '',
                language: '',
                audioPath: '',
                text: ''
            });
            this.renderReferences();
            this.updateDefaultRefDropdown();
        });

        document.getElementById('import-audio-btn').addEventListener('click', () => this.handleImportAudio());
        document.getElementById('save-avatar-btn').addEventListener('click', () => this.save());
    },

    renderList() {
        const body = document.getElementById('avatar-list-body');
        if (!Array.isArray(App.state.avatars)) {
            body.innerHTML = `<tr><td colspan="5">${t('Invalid voice list format')}</td></tr>`;
            return;
        }
        body.innerHTML = App.state.avatars.map(a => {
            if (!a) return '';
            const id = App.getProp(a, 'id');
            const name = App.getProp(a, 'name');
            const useV2 = App.getProp(a, 'useEngineV2');
            const refs = App.getProp(a, 'references') || [];

            const engineLabel = useV2 === true ? 'V2' : (useV2 === false ? 'V1' : 'AUTO');
            return `
            <tr>
                <td data-label="${t('ID')}"><code>${id}</code></td>
                <td data-label="${t('Name')}"><strong>${name || t('Unnamed')}</strong></td>
                <td data-label="${t('Engine')}"><span class="badge">${engineLabel}</span></td>
                <td data-label="${t('Reference Audio')}">${t('{count} refs', { count: refs.length })}</td>
                <td data-label="${t('Actions')}">
                    <button class="text-btn edit-btn" data-id="${id}">${t('Edit')}</button>
                    <button class="danger-btn delete-btn" data-id="${id}">${t('Delete')}</button>
                </td>
            </tr>
        `}).join('');

        body.querySelectorAll('.edit-btn').forEach(btn => {
            btn.addEventListener('click', () => this.openEditor(btn.dataset.id));
        });
        body.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', () => this.delete(btn.dataset.id));
        });
    },

    async openEditor(avatarId = null) {
        const isNew = !avatarId;
        const editor = document.getElementById('avatar-editor');
        if (avatarId) {
            const avatar = App.state.avatars.find(a => App.getProp(a, 'id') === avatarId);
            if (!avatar) {
                App.showToast(t('Voice not found: {id}', { id: avatarId }), 'error');
                return;
            }
            App.state.currentEditingAvatar = JSON.parse(JSON.stringify(avatar));
            document.getElementById('editor-title').textContent = t('Edit Voice: {name}', { name: App.getProp(avatar, 'name') || avatarId });
            document.getElementById('edit-avatar-is-new').value = 'false';
        } else {
            App.state.currentEditingAvatar = { id: '', name: '', description: '', useEngineV2: null, defaultReferenceId: 'default', references: [] };
            document.getElementById('editor-title').textContent = t('New Voice Configuration');
            document.getElementById('edit-avatar-is-new').value = 'true';
        }

        const form = document.getElementById('avatar-form');
        form.elements.Name.value = App.getProp(App.state.currentEditingAvatar, 'name') || '';
        form.elements.Description.value = App.getProp(App.state.currentEditingAvatar, 'description') || '';
        form.elements.UseEngineV2.value = String(App.getProp(App.state.currentEditingAvatar, 'useEngineV2'));

        await this.loadIdOptions(App.getProp(App.state.currentEditingAvatar, 'id'), isNew);

        // Events for ID and Engine change
        const engineSelect = form.elements.UseEngineV2;
        const onEngineChange = () => {
            this.loadIdOptions(document.getElementById('edit-avatar-id').value, isNew);
        };
        engineSelect.onchange = onEngineChange;

        const idSelect = document.getElementById('edit-avatar-id');
        idSelect.onchange = () => this.renderReferences();

        this.renderReferences();
        this.updateDefaultRefDropdown();

        editor.style.display = 'block';
        editor.scrollIntoView({ behavior: 'smooth' });
    },

    async loadIdOptions(currentId, isNew) {
        const idSelect = document.getElementById('edit-avatar-id');
        const engineVal = document.getElementById('avatar-form').elements.UseEngineV2.value;
        let useV2 = engineVal === 'null' ? !!App.getProp(App.state.config, 'useEngineV2') : engineVal === 'true';

        try {
            const resp = await fetch(`/api/tts/fs/avatars?useV2=${useV2}`, { cache: 'no-cache' });
            const entries = await resp.json();
            const registeredIds = new Set(App.state.avatars.map(a => App.getProp(a, 'id')));

            let options = isNew ? `<option value="">${t('— Select —')}</option>` : '';
            entries.forEach(e => {
                if (isNew && registeredIds.has(e.id) && e.id !== currentId) return;
                let label = e.id + (e.valid ? ' ✓' : (!e.hasModel ? t(' ⚠ Missing model') : t(' ⚠ Missing voice directory')));
                options += `<option value="${e.id}"${e.id === currentId ? ' selected' : ''}>${label}</option>`;
            });

            if (!isNew && currentId && !options.includes(`value="${currentId}"`)) {
                options = `<option value="${currentId}" selected>${currentId}</option>` + options;
            }
            idSelect.innerHTML = options;
            idSelect.disabled = !isNew;
        } catch (err) {
            idSelect.innerHTML = currentId ? `<option value="${currentId}" selected>${currentId}</option>` : `<option value="">${t('Load failed')}</option>`;
        }
    },

    renderReferences() {
        const container = document.getElementById('reference-list-container');
        const refs = App.getProp(App.state.currentEditingAvatar, 'references') || [];
        const avatarId = document.getElementById('edit-avatar-id').value;

        container.innerHTML = refs.map((r, idx) => {
            const rid = App.getProp(r, 'id') || '';
            const rname = App.getProp(r, 'name') || '';
            const rlang = App.getProp(r, 'language') || '';
            const rpath = App.getProp(r, 'audioPath') || '';
            const rtext = App.getProp(r, 'text') || '';
            const hintText = this.extractTextHint(rpath);

            return `
                <div class="reference-item" data-idx="${idx}">
                    <div class="field-row">
                        <div class="field"><label>ID</label><input type="text" value="${rid}" class="ref-id" placeholder="default"></div>
                        <div class="field"><label>Name</label><input type="text" value="${rname}" class="ref-name" placeholder="Display name"></div>
                        <div class="field">
                            <label>Language</label>
                            <select class="ref-lang styled-select">
                                <option value="" ${rlang === '' ? 'selected' : ''}>${t('Auto / unspecified')}</option>
                                <option value="zh" ${rlang === 'zh' ? 'selected' : ''}>${t('Chinese (zh)')}</option>
                                <option value="en" ${rlang === 'en' ? 'selected' : ''}>${t('English (en)')}</option>
                                <option value="ja" ${rlang === 'ja' ? 'selected' : ''}>${t('Japanese (ja)')}</option>
                            </select>
                        </div>
                    </div>
                    <div class="field"><label>Audio File (AudioPath)</label>
                        <select class="ref-path styled-select"><option value="${rpath}">${rpath || t('(Select file)')}</option></select>
                        <div class="ref-path-hint" title="${t('Click to copy filename')}">${hintText}</div>
                    </div>
                    <div class="field"><label>Text (Text)</label><input type="text" value="${rtext}" class="ref-text"></div>
                    <button type="button" class="danger-btn remove-ref-btn">${t('Remove')}</button>
                </div>`;
        }).join('');

        this.bindRefEvents(avatarId);
    },

    bindRefEvents(avatarId) {
        const container = document.getElementById('reference-list-container');
        if (avatarId) {
            fetch('/api/tts/fs/avatars/' + encodeURIComponent(avatarId) + '/references')
                .then(r => r.json())
                .then(files => {
                    container.querySelectorAll('.ref-path').forEach(sel => {
                        const val = sel.value;
                        sel.innerHTML = `<option value="">${t('(Select file)')}</option>` + files.map(f => `<option value="${f}"${f === val ? ' selected' : ''}>${f}</option>`).join('');
                        sel.onchange = () => {
                            const item = sel.closest('.reference-item');
                            const extracted = this.extractTextHint(sel.value);
                            item.querySelector('.ref-path-hint').textContent = extracted;
                            const textInput = item.querySelector('.ref-text');
                            if (extracted && !textInput.value.trim()) textInput.value = extracted;
                        };
                    });
                });
        }

        container.querySelectorAll('.ref-path-hint').forEach(hint => {
            hint.onclick = () => {
                const hintText = hint.textContent.trim();
                if (hintText) navigator.clipboard.writeText(hintText).then(() => App.showToast(t('Copied: {text}', { text: hintText })));
            };
        });

        container.querySelectorAll('.ref-id').forEach(input => {
            input.oninput = () => {
                const idx = input.closest('.reference-item').dataset.idx;
                App.state.currentEditingAvatar.references[idx].id = input.value;
                this.updateDefaultRefDropdown();
            };
        });

        container.querySelectorAll('.remove-ref-btn').forEach(btn => {
            btn.onclick = () => {
                this.syncFromDOM();
                const idx = btn.closest('.reference-item').dataset.idx;
                App.state.currentEditingAvatar.references.splice(idx, 1);
                this.renderReferences();
                this.updateDefaultRefDropdown();
            };
        });
    },

    syncFromDOM() {
        const container = document.getElementById('reference-list-container');
        container.querySelectorAll('.reference-item').forEach(item => {
            const idx = item.dataset.idx;
            const ref = App.state.currentEditingAvatar.references[idx];
            if (!ref) return;
            ref.id = item.querySelector('.ref-id').value;
            ref.name = item.querySelector('.ref-name').value;
            ref.language = item.querySelector('.ref-lang').value;
            ref.audioPath = item.querySelector('.ref-path').value;
            ref.text = item.querySelector('.ref-text').value;

            // Compat for PascalCase (ensure consistency for saving)
            ref.Id = ref.id; ref.Name = ref.name; ref.Language = ref.language;
            ref.AudioPath = ref.audioPath; ref.Text = ref.text;
        });
    },

    updateDefaultRefDropdown() {
        const select = document.getElementById('avatar-form').elements.DefaultReferenceId;
        const refs = App.state.currentEditingAvatar.references || [];
        const current = App.getProp(App.state.currentEditingAvatar, 'defaultReferenceId') || 'default';
        select.innerHTML = refs.length === 0 ? '<option value="default">default</option>' :
            refs.map(r => `<option value="${r.id}"${r.id === current ? ' selected' : ''}>${r.name || r.id}</option>`).join('');
    },

    extractTextHint(filename) {
        if (!filename) return '';
        return filename.replace(/\.[^.]+$/, '').replace(/^【[^】]*】/, '').trim();
    },

    async save() {
        this.syncFromDOM();
        const form = document.getElementById('avatar-form');
        const elements = form.elements;

        const updated = {
            id: elements['Id'].value,
            name: elements['Name'].value,
            description: elements['Description'].value,
            useEngineV2: elements['UseEngineV2'].value === 'null' ? null : (elements['UseEngineV2'].value === 'true'),
            defaultReferenceId: elements['DefaultReferenceId'].value || 'default',
            references: App.state.currentEditingAvatar.references
        };

        App.showLoading(t('Saving voice and reloading configuration...'));
        try {
            const resp = await fetch('/api/tts/avatars', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updated)
            });
            if (resp.ok) {
                App.showToast(t('Voice saved successfully'));
                document.getElementById('avatar-editor').style.display = 'none';
                await App.fetchAll();
            } else {
                const res = await resp.json();
                throw new Error(res.message || t('Save failed'));
            }
        } catch (e) { App.showToast(t('Save error: {message}', { message: e.message }), 'error'); }
        finally { App.hideLoading(); }
    },

    async handleImportAudio() {
        const avatarId = document.getElementById('edit-avatar-id').value;
        if (!avatarId) return App.showToast(t('Select a voice ID first'), 'error');

        Converter.openFilePicker('*.wav', async (filePath) => {
            try {
                App.showLoading(t('Importing audio file...'));
                const targetDir = `resources/avatars/${avatarId}/references`;
                const resp = await fetch(`/api/tts/fs/copy-file?sourcePath=${encodeURIComponent(filePath)}&targetDir=${encodeURIComponent(targetDir)}`, { method: 'POST' });
                if (!resp.ok) throw new Error(t('File copy failed'));
                
                App.hideLoading();
                App.showToast(t('File imported into the references directory. Refreshing list...'));
                
                // Refresh file list (rebind dropdown)
                this.bindRefEvents(avatarId);
            } catch (err) { 
                App.hideLoading(); 
                App.showToast(t('Import failed: {message}', { message: err.message }), 'error'); 
            }
        });
    },

    async delete(id) {
        if (!confirm(t('Delete the configuration for voice [{id}]? This does not remove physical model files.', { id }))) return;
        App.showLoading(t('Deleting voice...'));
        try {
            const resp = await fetch(`/api/tts/avatars/${id}`, { method: 'DELETE' });
            if (resp.ok) {
                App.showToast(t('Deleted successfully'));
                await App.fetchAll();
            } else {
                const res = await resp.json();
                throw new Error(res.message || t('Delete failed'));
            }
        } catch (e) { App.showToast(t('Error: {message}', { message: e.message }), 'error'); }
        finally { App.hideLoading(); }
    }
};
