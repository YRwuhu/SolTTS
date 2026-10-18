// ============================================================
// SolTTS WebUI — Dashboard Module
// Dashboard rendering and runtime status updates
// ============================================================

const Dashboard = {
    updateStatus() {
        const dot = document.querySelector('.status-dot');
        const label = document.getElementById('runtime-status');
        const statusVal = App.state.status ? (App.getProp(App.state.status, 'status') || '') : '';
        const isOnline = statusVal.toLowerCase() === 'online';
        const engineName = App.state.info ? (App.getProp(App.state.info, 'engine') || '') : '';

        dot.classList.toggle('online', isOnline);
        if (isOnline) {
            label.textContent = engineName ? engineName + ' — ' + t('Engine Online') : t('Engine Online');
        } else {
            label.textContent = t('Engine Offline');
        }
    },

    update() {
        this.updateStatus();
        if (!App.state.info) return;
        document.getElementById('stat-sampling-rate').textContent = App.getProp(App.state.info, 'samplingRate') || '--';
        document.getElementById('stat-engine-type').textContent = App.getProp(App.state.info, 'engine') || '--';
        document.getElementById('stat-avatar-count').textContent = App.state.avatars.length;
        document.getElementById('stat-default-id').textContent = App.getProp(App.state.config, 'defaultAvatarId') || '--';
        document.getElementById('stat-resources-dir').textContent = App.getProp(App.state.config, 'resourcesDir') || '--';
        document.getElementById('stat-v2-enabled').textContent = App.getProp(App.state.config, 'useEngineV2') ? t('Enabled') : t('Disabled (V1)');
    }
};
