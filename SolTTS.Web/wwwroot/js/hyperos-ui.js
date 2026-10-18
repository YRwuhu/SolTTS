// ============================================================
// SolTTS WebUI — HyperOS UI Enhancements
// Ripple feedback, theme sync, breadcrumbs, quick actions
// Design language and tokens ported from miuix-console (Apache-2.0)
// ============================================================

const HyperOS = {
    tabs: {
        dashboard: 'Dashboard',
        avatars: 'Voice Library',
        config: 'Control Center',
        playground: 'Synthesis Lab',
        converter: 'Model Converter'
    },

    init() {
        this.bindRipple();
        this.bindQuickActions();
        this.bindStressToggle();
        this.wrapSelects();
        this.syncThemeMeta();
    },

    // Wrap every <select> in .select-wrap so the arrow can be drawn by CSS.
    // JS-rendered markup (avatar editor, reference rows) is handled by the observer.
    wrapSelects() {
        const wrap = (select) => {
            if (select.closest('.select-wrap')) return;
            const parent = select.parentNode;
            if (!parent) return;
            const wrapper = document.createElement('span');
            wrapper.className = 'select-wrap';
            parent.insertBefore(wrapper, select);
            wrapper.appendChild(select);
        };

        document.querySelectorAll('select').forEach(wrap);

        const observer = new MutationObserver(mutations => {
            mutations.forEach(m => m.addedNodes.forEach(node => {
                if (node.nodeType !== 1) return;
                if (node.tagName === 'SELECT') wrap(node);
                node.querySelectorAll?.('select').forEach(wrap);
            }));
        });
        observer.observe(document.body, { childList: true, subtree: true });
    },

    // Miuix-style pointer ripple for [data-ripple] controls
    bindRipple() {
        document.addEventListener('pointerdown', (e) => {
            const el = e.target.closest('[data-ripple]');
            if (!el || el.disabled) return;

            const rect = el.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height) * 2;
            const fx = document.createElement('span');

            fx.className = 'mx-ripple-fx';
            fx.style.width = size + 'px';
            fx.style.height = size + 'px';
            fx.style.left = (e.clientX - rect.left - size / 2) + 'px';
            fx.style.top = (e.clientY - rect.top - size / 2) + 'px';

            el.appendChild(fx);
            setTimeout(() => fx.remove(), 620);
        });
    },

    bindQuickActions() {
        document.querySelectorAll('[data-goto]').forEach(btn => {
            btn.addEventListener('click', () => {
                if (typeof App !== 'undefined') App.switchTab(btn.dataset.goto);
            });
        });
    },

    // The stress panel toggles inline, so keep the arrow direction in sync
    bindStressToggle() {
        const toggle = document.getElementById('stress-test-toggle');
        const panel = document.getElementById('stress-test-panel');
        if (!toggle || !panel) return;

        const observer = new MutationObserver(() => {
            toggle.classList.toggle('open', panel.style.display !== 'none');
        });
        observer.observe(panel, { attributes: true, attributeFilter: ['style'] });
    },

    updateChrome(tabId) {
        const label = t(this.tabs[tabId] || 'SolTTS');
        const crumb = document.getElementById('breadcrumb-title');
        if (crumb) crumb.textContent = label;
        document.title = t('SolTTS Control Center') + ' — ' + label;
    },

    syncThemeMeta() {
        const meta = document.querySelector('meta[name="theme-color"]');
        const apply = () => {
            if (!meta) return;
            const dark = document.documentElement.dataset.theme !== 'light';
            meta.content = dark ? '#0C0C0C' : '#F7F7F7';
        };
        apply();
        window.addEventListener('mx-theme-change', apply);
    }
};

document.addEventListener('DOMContentLoaded', () => {
    if (typeof I18N !== 'undefined') I18N.init();
    HyperOS.init();
});
