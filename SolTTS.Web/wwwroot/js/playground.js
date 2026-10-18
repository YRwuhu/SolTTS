// ============================================================
// SolTTS WebUI — Playground Module
// Synthesis lab, streaming playback, concurrency stress test
// ============================================================

const Playground = {
    audioContext: null,
    pcmPlayer: null,

    init() {
        document.getElementById('pg-play-btn').addEventListener('click', () => {
            const isStream = document.getElementById('pg-stream-toggle')?.checked;
            if (isStream) this.playStream();
            else this.playStandard();
        });

        document.getElementById('pg-speed').addEventListener('input', (e) => {
            document.getElementById('pg-speed-val').textContent = e.target.value;
        });

        const emoAlpha = document.getElementById('pg-emotion-alpha');
        if (emoAlpha) {
            emoAlpha.addEventListener('input', (e) => {
                document.getElementById('pg-emotion-alpha-val').textContent =
                    parseFloat(e.target.value).toFixed(2);
            });
        }

        // Max 2 langs
        const container = document.getElementById('pg-langs');
        if (container) {
            container.addEventListener('change', (e) => {
                if (e.target.type === 'checkbox') {
                    if (container.querySelectorAll('input:checked').length > 2) {
                        App.showToast(t('Select at most two languages'), 'warn');
                        e.target.checked = false;
                    }
                }
            });
        }

        // Stress Test toggle
        const stressToggle = document.getElementById('stress-test-toggle');
        if (stressToggle) {
            stressToggle.onclick = () => {
                const panel = document.getElementById('stress-test-panel');
                panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
            };
        }
        document.getElementById('run-stress-btn')?.addEventListener('click', () => this.runStressTest());
    },

    updateAvatars() {
        const select = document.getElementById('pg-avatar-select');
        const defaultId = App.getProp(App.state.config, 'defaultAvatarId');
        
        select.innerHTML = App.state.avatars.map(a => {
            const id = App.getProp(a, 'id');
            const name = App.getProp(a, 'name');
            return `<option value="${id}"${id === defaultId ? ' selected' : ''}>${name || id} (${id})</option>`;
        }).join('');

        const updateRefs = () => {
            const avatarId = select.value;
            const avatar = App.state.avatars.find(a => App.getProp(a, 'id') === avatarId);
            const refSelect = document.getElementById('pg-ref-id');
            const refs = App.getProp(avatar, 'references') || [];
            refSelect.innerHTML = `<option value="">${t('Use default')}</option>` + 
                refs.map(r => `<option value="${r.id}">${r.name || r.id}</option>`).join('');

            // Update V2 params visibility
            const globalV2 = !!App.getProp(App.state.config, 'useEngineV2');
            const isV2 = avatar ? (App.getProp(avatar, 'useEngineV2') ?? globalV2) : globalV2;
            document.getElementById('v2-params').style.display = isV2 ? 'block' : 'none';
        };

        select.onchange = updateRefs;
        updateRefs();
    },

    getParams() {
        const avatarId = document.getElementById('pg-avatar-select').value;
        const avatar = App.state.avatars.find(a => App.getProp(a, 'id') === avatarId);
        const globalV2 = !!App.getProp(App.state.config, 'useEngineV2');
        const isV2 = avatar ? (App.getProp(avatar, 'useEngineV2') ?? globalV2) : globalV2;
        const selectedLangs = Array.from(document.querySelectorAll('#pg-langs input:checked')).map(i => i.value);

        const params = {
            text: document.getElementById('pg-text').value,
            avatarId: avatarId,
            referenceId: document.getElementById('pg-ref-id').value || null,
            speed: parseFloat(document.getElementById('pg-speed').value),
            languages: selectedLangs.length > 0 ? selectedLangs : null
        };

        // Emotion prompting (supported by the V1 engine only; the backend silently ignores it on V2)
        const emotionId = document.getElementById('pg-emotion')?.value || null;
        if (emotionId) {
            params.emotionId = emotionId;
            params.emotionAlpha = parseFloat(document.getElementById('pg-emotion-alpha').value);
        }

        if (isV2) {
            params.temperature = parseFloat(document.getElementById('pg-temp').value);
            params.noiseScale = parseFloat(document.getElementById('pg-noise').value);
            params.topK = parseInt(document.getElementById('pg-topk').value);
        }
        return params;
    },

    async playStandard() {
        const params = this.getParams();
        if (!params.text) return App.showToast(t('Enter text to synthesize'), 'error');

        document.getElementById('pg-play-btn').disabled = true;
        document.getElementById('playing-status').style.display = 'block';
        document.getElementById('synth-perf-panel').style.display = 'none';
        document.getElementById('pg-download-area').innerHTML = '';

        const startTime = performance.now();
        try {
            const resp = await fetch('/api/tts/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });

            if (!resp.ok) throw new Error(t('Synthesis failed'));

            this.updateMetrics(resp, performance.now() - startTime);

            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            new Audio(url).play();

            // Download link
            const ts = new Date().toTimeString().slice(0, 8).replace(/:/g, '');
            document.getElementById('pg-download-area').innerHTML =
                `<a href="${url}" download="tts_${ts}.wav" class="download-link">💾 ${t('Download synthesized audio (WAV)')}</a>`;

            App.showToast(t('Synthesis succeeded, playing audio'));
        } catch (e) { App.showToast(t('Synthesis error: {message}', { message: e.message }), 'error'); }
        finally {
            document.getElementById('pg-play-btn').disabled = false;
            document.getElementById('playing-status').style.display = 'none';
        }
    },

    async playStream() {
        const params = this.getParams();
        if (!params.text) return App.showToast(t('Enter text to synthesize'), 'error');

        document.getElementById('pg-play-btn').disabled = true;
        document.getElementById('playing-status').style.display = 'block';
        document.getElementById('playing-status').textContent = t('Initializing streaming audio...');
        document.getElementById('pg-download-area').innerHTML = '';
        document.getElementById('synth-perf-panel').style.display = 'none';

        try {
            const query = new URLSearchParams();
            for (let k in params) {
                if (params[k] !== null) {
                   if (Array.isArray(params[k])) params[k].forEach(v => query.append(k, v));
                   else query.append(k, params[k]);
                }
            }

            const startTime = performance.now();
            const response = await fetch('/api/tts/predict-stream?' + query.toString());
            if (!response.ok) throw new Error(t('Streaming request failed'));

            // Read the actual sample rate from the response headers to avoid pitch artifacts from hardcoding
            const sampleRate = parseInt(response.headers.get('X-Audio-Sample-Rate')) || 32000;

            // (Re)create the AudioContext if the sample rate changed or on first use
            if (!this.audioContext || this.audioContext.sampleRate !== sampleRate) {
                if (this.audioContext) this.audioContext.close();
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate });
            }
            if (this.audioContext.state === 'suspended') await this.audioContext.resume();

            const reader = response.body.getReader();
            let firstChunkTime = 0;
            let firstChunk = true;
            let totalChunks = 0;
            let totalSamples = 0;
            let nextStartTime = this.audioContext.currentTime;
            const allChunks = [];

            // PCM float32 alignment buffer: ReadableStream chunks may be split
            // outside 4-byte boundaries, causing Float32Array construction failures or misaligned data
            let residual = new Uint8Array(0);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                // Concatenate residual bytes with the new data
                let combined;
                if (residual.length > 0) {
                    combined = new Uint8Array(residual.length + value.length);
                    combined.set(residual);
                    combined.set(value, residual.length);
                } else {
                    combined = value;
                }

                // Take the 4-byte aligned portion; keep the remainder in residual
                const alignedLen = combined.length - (combined.length % 4);
                residual = combined.slice(alignedLen);

                if (alignedLen === 0) continue;

                if (firstChunk) {
                    firstChunkTime = performance.now() - startTime;
                    document.getElementById('perf-fbl').textContent = Math.round(firstChunkTime);
                    document.getElementById('synth-perf-panel').style.display = 'flex';
                    firstChunk = false;
                }

                // Safely construct a Float32Array: extract each sample with DataView to ensure correct byte order
                const sampleCount = alignedLen / 4;
                const floatData = new Float32Array(sampleCount);
                const dataView = new DataView(combined.buffer, combined.byteOffset, alignedLen);
                for (let i = 0; i < sampleCount; i++) {
                    floatData[i] = dataView.getFloat32(i * 4, true); // little-endian
                }

                allChunks.push(floatData);
                totalSamples += sampleCount;

                const audioBuffer = this.audioContext.createBuffer(1, floatData.length, sampleRate);
                audioBuffer.getChannelData(0).set(floatData);

                const source = this.audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(this.audioContext.destination);
                const startAt = Math.max(nextStartTime, this.audioContext.currentTime);
                source.start(startAt);
                nextStartTime = startAt + audioBuffer.duration;

                totalChunks++;
                document.getElementById('playing-status').textContent = t('Streaming ({hz}Hz): {chunks} chunks received...', { hz: sampleRate, chunks: totalChunks });
            }

            // Streaming performance metrics
            const totalElapsed = performance.now() - startTime;
            const audioDuration = totalSamples / sampleRate;
            const rtf = (totalElapsed / 1000) / audioDuration;
            const speed = (audioDuration / (totalElapsed / 1000)).toFixed(2);
            document.getElementById('perf-fbl-label').textContent = t('First-Chunk Latency');
            document.getElementById('perf-fbl-unit').textContent = t('ms');
            document.getElementById('perf-speed').textContent = speed;
            document.getElementById('perf-speed-unit').textContent = 'x';
            document.getElementById('perf-rtf').textContent = rtf.toFixed(3);

            // Assemble a WAV blob for download
            const wavBlob = this.pcmToWav(allChunks, totalSamples, sampleRate);
            const wavUrl = URL.createObjectURL(wavBlob);
            const ts = new Date().toTimeString().slice(0, 8).replace(/:/g, '');
            document.getElementById('pg-download-area').innerHTML =
                `<a href="${wavUrl}" download="tts_stream_${ts}.wav" class="download-link">💾 ${t('Download streamed audio (WAV, {duration}s)', { duration: audioDuration.toFixed(1) })}</a>`;

            App.showToast(t('Streaming synthesis complete'));
        } catch (e) { App.showToast(t('Streaming error: {message}', { message: e.message }), 'error'); }
        finally {
            document.getElementById('pg-play-btn').disabled = false;
            document.getElementById('playing-status').style.display = 'none';
            document.getElementById('playing-status').textContent = t('Processing audio stream...');
        }
    },

    updateMetrics(resp, fbl) {
        const rtf = resp.headers.get('X-RTF');
        const totalTime = resp.headers.get('X-Synthesis-Time');
        const tokenCount = resp.headers.get('X-Token-Count');

        // Standard mode labels
        document.getElementById('perf-fbl-label').textContent = t('Generation Time');
        document.getElementById('perf-fbl-unit').textContent = t('ms');
        document.getElementById('perf-speed-unit').textContent = t('tokens/s');
        document.getElementById('perf-fbl').textContent = Math.round(fbl);
        if (rtf) document.getElementById('perf-rtf').textContent = rtf;
        
        if (tokenCount && totalTime) {
            const speed = (parseInt(tokenCount) / (parseFloat(totalTime) / 1000.0)).toFixed(1);
            document.getElementById('perf-speed').textContent = speed;
        } else if (rtf) {
            document.getElementById('perf-speed').textContent = (1.0 / parseFloat(rtf)).toFixed(2);
        }
        document.getElementById('synth-perf-panel').style.display = 'flex';
    },

    /** Assemble Float32 PCM chunks into a WAV Blob (16-bit PCM) */
    pcmToWav(chunks, totalSamples, sampleRate) {
        const buffer = new ArrayBuffer(44 + totalSamples * 2);
        const view = new DataView(buffer);
        const writeStr = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
        const dataSize = totalSamples * 2;
        writeStr(0, 'RIFF');
        view.setUint32(4, 36 + dataSize, true);
        writeStr(8, 'WAVE');
        writeStr(12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeStr(36, 'data');
        view.setUint32(40, dataSize, true);
        let offset = 44;
        for (const chunk of chunks) {
            for (let i = 0; i < chunk.length; i++) {
                const s = Math.max(-1, Math.min(1, chunk[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                offset += 2;
            }
        }
        return new Blob([buffer], { type: 'audio/wav' });
    },

    // --- Stress Test ---

    async runStressTest() {
        const concurrency = parseInt(document.getElementById('stress-concurrency').value) || 1;
        const repeats = parseInt(document.getElementById('stress-repeats').value) || 1;
        
        // Use the dedicated stress-test text if provided; otherwise fall back to the playground text
        const stressText = document.getElementById('stress-text')?.value?.trim();
        const baseText = stressText || document.getElementById('pg-text').value || "测试并发。";
        
        // Split multi-line text (one test case per line, used round-robin)
        const textLines = baseText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
        
        // Support rotating through multiple voices
        const stressAvatarMode = document.getElementById('stress-avatar-mode')?.value || 'current';
        let avatarIds = [];
        if (stressAvatarMode === 'all') {
            avatarIds = App.state.avatars.map(a => App.getProp(a, 'id'));
        } else {
            avatarIds = [document.getElementById('pg-avatar-select').value];
        }

        // Reference audio
        const stressRefId = document.getElementById('stress-ref-id')?.value || '';
        
        const resultsTable = document.getElementById('stress-results-body');
        resultsTable.innerHTML = '';
        document.getElementById('stress-stats').innerHTML = t('Initializing...');
        document.getElementById('run-stress-btn').disabled = true;
        
        const tasks = [];
        let completed = 0;
        let successful = 0;
        let totalLatency = 0;
        let minLatency = Infinity;
        let maxLatency = 0;
        const startTime = performance.now();

        for (let i = 0; i < repeats; i++) {
            const text = textLines[i % textLines.length];
            const avatarId = avatarIds[i % avatarIds.length];

            tasks.push(async (id) => {
                const row = document.createElement('tr');
                row.innerHTML = `<td data-label="${t('ID')}">#${id}</td><td data-label="${t('Voice')}">${avatarId}</td><td data-label="${t('Status')}" class="status-pending">${t('Running...')}</td><td data-label="${t('Latency')}">-</td><td data-label="${t('Duration')}">-</td><td data-label="${t('Download')}">-</td>`;
                resultsTable.prepend(row);
                
                const body = { text, avatarId };
                if (stressRefId) body.referenceId = stressRefId;
                
                const reqStart = performance.now();
                try {
                    const resp = await fetch('/api/tts/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(body)
                    });
                    const duration = performance.now() - reqStart;
                    if (resp.ok) {
                        successful++;
                        totalLatency += duration;
                        if (duration < minLatency) minLatency = duration;
                        if (duration > maxLatency) maxLatency = duration;
                        row.cells[2].textContent = t('Success');
                        row.cells[2].className = 'status-success';
                        row.cells[3].textContent = Math.round(duration) + 'ms';
                        row.cells[4].textContent = (resp.headers.get('X-Audio-Duration') || '-') + 's';
                        // Download link
                        const blob = await resp.blob();
                        const url = URL.createObjectURL(blob);
                        row.cells[5].innerHTML = `<a href="${url}" download="stress_${id}.wav" title="Download">💾</a>`;
                    } else {
                        row.cells[2].textContent = t('Failed ({status})', { status: resp.status });
                        row.cells[2].className = 'status-fail';
                        row.cells[3].textContent = Math.round(duration) + 'ms';
                    }
                } catch (e) {
                    row.cells[2].textContent = t('Error');
                    row.cells[2].className = 'status-fail';
                    row.cells[3].textContent = e.message;
                }
                completed++;
                this.updateStressStats(startTime, completed, repeats, successful, totalLatency, minLatency, maxLatency);
            });
        }

        const limit = async (concurrency, tasks) => {
            const pool = new Set();
            for (let i = 0; i < tasks.length; i++) {
                const promise = tasks[i](i + 1);
                pool.add(promise);
                promise.then(() => pool.delete(promise));
                if (pool.size >= concurrency) await Promise.race(pool);
            }
            await Promise.all(pool);
        };

        App.showToast(t('Stress test started: concurrency {concurrency}, total {total}, {texts} text(s), {voices} voice(s)', { concurrency, total: repeats, texts: textLines.length, voices: avatarIds.length }));
        await limit(concurrency, tasks);
        document.getElementById('run-stress-btn').disabled = false;
        App.showToast(t('Stress test complete'));
    },

    updateStressStats(startTime, completed, total, successful, totalLatency, minLat, maxLat) {
        const elapsed = (performance.now() - startTime) / 1000;
        const avgLat = successful > 0 ? (totalLatency / successful).toFixed(0) : '-';
        const qps = (completed / elapsed).toFixed(2);
        const failed = completed - successful;
        const minStr = minLat < Infinity ? Math.round(minLat) : '-';
        const maxStr = maxLat > 0 ? Math.round(maxLat) : '-';
        document.getElementById('stress-stats').innerHTML = 
            `<span>${t('Progress: {done}/{total}', { done: completed, total })}</span> | ` +
            `<span style="color:var(--online-color)">${t('Success: {count}', { count: successful })}</span>` +
            (failed > 0 ? ` | <span style="color:#ef4444">${t('Failed: {count}', { count: failed })}</span>` : '') +
            ` | ${t('Avg: {avg}ms | Min: {min}ms | Max: {max}ms', { avg: avgLat, min: minStr, max: maxStr })}` +
            ` | ${t('QPS: {qps} | Total: {total}s', { qps, total: elapsed.toFixed(1) }).replace('{qps}', '<strong>' + qps + '</strong>')}`;
    }
};
