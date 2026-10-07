/**
 * Fashion-MNIST CNN Image Classifier
 * Frontend Controller for Camera, Uploads, API Inference & Metric Visuals
 */

document.addEventListener('DOMContentLoaded', () => {
    // State variables
    let currentImageBase64 = null;
    let cameraStream = null;
    let isCameraActive = false;

    // DOM Elements
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Camera DOM
    const cameraVideo = document.getElementById('camera-video');
    const btnStartCamera = document.getElementById('btn-start-camera');
    const btnStopCamera = document.getElementById('btn-stop-camera');
    const btnSnapPhoto = document.getElementById('btn-snap-photo');
    const cameraInactiveMsg = document.getElementById('camera-inactive-msg');

    // Samples DOM
    const samplesGrid = document.getElementById('samples-grid');

    // Preview & Action DOM
    const previewDisplayBox = document.getElementById('preview-display-box');
    const previewPlaceholder = document.getElementById('preview-placeholder');
    const previewActiveWrap = document.getElementById('preview-active-wrap');
    const previewImg = document.getElementById('preview-img');
    const removeImgBtn = document.getElementById('remove-img-btn');
    const predictBtn = document.getElementById('predict-btn');
    const predictBtnText = document.getElementById('predict-btn-text');
    const invertModeSelect = document.getElementById('invert-mode-select');

    // Results DOM
    const loadingState = document.getElementById('loading-state');
    const resultsContent = document.getElementById('results-content');
    const resCategory = document.getElementById('res-category');
    const resDescription = document.getElementById('res-description');
    const resConfidence = document.getElementById('res-confidence');
    const confBadgeCircle = document.getElementById('conf-badge-circle');
    const tensorPreprocessedImg = document.getElementById('tensor-preprocessed-img');
    const inversionStatusText = document.getElementById('inversion-status-text');
    const probBarsList = document.getElementById('prob-bars-list');
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');

    // Hero buttons
    const heroUploadBtn = document.getElementById('hero-upload-btn');
    const heroCameraBtn = document.getElementById('hero-camera-btn');

    // -------------------------------------------------------------
    // Tab Switching
    // -------------------------------------------------------------
    function switchTab(targetTabName) {
        tabButtons.forEach(btn => {
            if (btn.getAttribute('data-tab') === targetTabName) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        tabContents.forEach(content => {
            if (content.id === `content-${targetTabName}`) {
                content.classList.add('active');
            } else {
                content.classList.remove('active');
            }
        });

        // If leaving camera tab, stop camera to save resources
        if (targetTabName !== 'camera' && isCameraActive) {
            stopCamera();
        }
        // If entering camera tab and not running, prompt start
        if (targetTabName === 'camera' && !isCameraActive) {
            startCamera();
        }
    }

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');
            switchTab(tabName);
        });
    });

    // Hero buttons binding
    if (heroUploadBtn) {
        heroUploadBtn.addEventListener('click', () => {
            document.getElementById('predict-section').scrollIntoView({ behavior: 'smooth' });
            switchTab('upload');
            fileInput.click();
        });
    }

    if (heroCameraBtn) {
        heroCameraBtn.addEventListener('click', () => {
            document.getElementById('predict-section').scrollIntoView({ behavior: 'smooth' });
            switchTab('camera');
        });
    }

    // -------------------------------------------------------------
    // Image Selection & Preview Handling
    // -------------------------------------------------------------
    function setImage(base64Data, label = 'Ready for Inference') {
        currentImageBase64 = base64Data;
        previewImg.src = base64Data;
        
        previewPlaceholder.style.display = 'none';
        previewActiveWrap.style.display = 'flex';
        removeImgBtn.style.display = 'inline-flex';
        predictBtn.disabled = false;
        
        const badge = document.getElementById('preview-badge');
        if (badge) badge.textContent = label;

        // Clear error alert if any
        hideError();
    }

    function clearImage() {
        currentImageBase64 = null;
        previewImg.src = '';
        fileInput.value = '';
        
        previewPlaceholder.style.display = 'block';
        previewActiveWrap.style.display = 'none';
        removeImgBtn.style.display = 'none';
        predictBtn.disabled = true;

        // Hide results when image is removed
        resultsContent.style.display = 'none';
        hideError();

        // Deselect sample buttons if any
        document.querySelectorAll('.sample-item-btn').forEach(btn => btn.classList.remove('selected'));
    }

    removeImgBtn.addEventListener('click', clearImage);

    // -------------------------------------------------------------
    // File Upload via Dropzone or Click
    // -------------------------------------------------------------
    dropzone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleUploadedFile(e.target.files[0]);
        }
    });

    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleUploadedFile(e.dataTransfer.files[0]);
        }
    });

    function handleUploadedFile(file) {
        if (!file.type.startsWith('image/')) {
            showError('Selected file is not an image. Please upload a valid PNG, JPG, or WEBP file.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            setImage(e.target.result, `File: ${file.name}`);
        };
        reader.onerror = () => {
            showError('Failed to read the selected file.');
        };
        reader.readAsDataURL(file);
    }

    // -------------------------------------------------------------
    // Camera Integration (getUserMedia)
    // -------------------------------------------------------------
    async function startCamera() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showError('Camera access is not supported by this browser.');
            return;
        }

        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraVideo.srcObject = cameraStream;
            cameraVideo.play();

            isCameraActive = true;
            cameraInactiveMsg.style.display = 'none';
            btnStopCamera.style.display = 'inline-flex';
            btnSnapPhoto.disabled = false;
        } catch (err) {
            console.error('Camera permission error:', err);
            isCameraActive = false;
            cameraInactiveMsg.style.display = 'flex';
            btnSnapPhoto.disabled = true;
            showError(`Could not access camera: ${err.message || 'Permission denied'}. You can still upload files.`);
        }
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        cameraVideo.srcObject = null;
        isCameraActive = false;
        cameraInactiveMsg.style.display = 'flex';
        btnStopCamera.style.display = 'none';
        btnSnapPhoto.disabled = true;
    }

    btnStartCamera.addEventListener('click', startCamera);
    btnStopCamera.addEventListener('click', stopCamera);

    btnSnapPhoto.addEventListener('click', () => {
        if (!isCameraActive || !cameraVideo.videoWidth) return;

        // Capture frame from video onto offscreen canvas
        const canvas = document.createElement('canvas');
        canvas.width = cameraVideo.videoWidth;
        canvas.height = cameraVideo.videoHeight;
        const ctx = canvas.getContext('2d');
        
        // Un-mirror snapshot
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
        ctx.drawImage(cameraVideo, 0, 0, canvas.width, canvas.height);

        const dataUrl = canvas.toDataURL('image/jpeg', 0.95);
        setImage(dataUrl, 'Captured from Camera');

        // Optional: stop camera after capture to be clean
        stopCamera();
    });

    // -------------------------------------------------------------
    // Benchmark Samples & Real Studio Photos Toggle & Fetch
    // -------------------------------------------------------------
    let benchmarkSamples = [];
    let realStudioSamples = [];
    let currentSampleType = 'bench'; // 'bench' or 'real'

    const btnSamplesBench = document.getElementById('btn-samples-bench');
    const btnSamplesReal = document.getElementById('btn-samples-real');
    const samplesHintText = document.getElementById('samples-hint-text');

    if (btnSamplesBench && btnSamplesReal) {
        btnSamplesBench.addEventListener('click', () => {
            currentSampleType = 'bench';
            btnSamplesBench.classList.add('active');
            btnSamplesReal.classList.remove('active');
            if (samplesHintText) {
                samplesHintText.innerHTML = '<i data-lucide="info"></i> Click any 28×28 dataset benchmark sample to evaluate the model instantly:';
                if (window.lucide) window.lucide.createIcons();
            }
            renderSamples(benchmarkSamples, 'Dataset');
        });

        btnSamplesReal.addEventListener('click', () => {
            currentSampleType = 'real';
            btnSamplesReal.classList.add('active');
            btnSamplesBench.classList.remove('active');
            if (samplesHintText) {
                samplesHintText.innerHTML = '<i data-lucide="info"></i> Click any realistic studio clothing photo to evaluate background isolation & CNN inference:';
                if (window.lucide) window.lucide.createIcons();
            }
            renderSamples(realStudioSamples, 'Studio Photo');
        });
    }

    async function loadSamples() {
        try {
            const [respBench, respReal] = await Promise.all([
                fetch('/api/samples'),
                fetch('/api/real_samples')
            ]);
            
            if (respBench.ok) {
                benchmarkSamples = await respBench.json();
            }
            if (respReal.ok) {
                realStudioSamples = await respReal.json();
            }

            if (currentSampleType === 'bench' && benchmarkSamples.length > 0) {
                renderSamples(benchmarkSamples, 'Dataset');
            } else if (realStudioSamples.length > 0) {
                renderSamples(realStudioSamples, 'Studio Photo');
            }
        } catch (e) {
            console.log('Sample catalog fetch error:', e);
        }
    }

    function renderSamples(samples, typePrefix = 'Benchmark') {
        if (!samplesGrid) return;
        samplesGrid.innerHTML = '';
        if (!Array.isArray(samples) || samples.length === 0) {
            samplesGrid.innerHTML = '<div class="samples-loading">No samples available.</div>';
            return;
        }

        samples.forEach((sample) => {
            const itemBtn = document.createElement('button');
            itemBtn.className = 'sample-item-btn';
            itemBtn.type = 'button';
            itemBtn.innerHTML = `
                <img src="${sample.url}" alt="${sample.label}" class="sample-img-thumb">
                <span class="sample-lbl">${sample.label}</span>
            `;

            itemBtn.addEventListener('click', async () => {
                document.querySelectorAll('.sample-item-btn').forEach(b => b.classList.remove('selected'));
                itemBtn.classList.add('selected');

                try {
                    const imgResponse = await fetch(sample.url);
                    const blob = await imgResponse.blob();
                    const reader = new FileReader();
                    reader.onload = (e) => {
                        setImage(e.target.result, `${typePrefix}: ${sample.label}`);
                    };
                    reader.readAsDataURL(blob);
                } catch (e) {
                    setImage(sample.url, `${typePrefix}: ${sample.label}`);
                }
            });

            samplesGrid.appendChild(itemBtn);
        });
    }

    // -------------------------------------------------------------
    // Prediction Request to Flask Backend
    // -------------------------------------------------------------
    predictBtn.addEventListener('click', async () => {
        if (!currentImageBase64) {
            showError('Please upload or capture an image first.');
            return;
        }

        hideError();
        resultsContent.style.display = 'none';
        loadingState.style.display = 'block';
        predictBtn.disabled = true;
        predictBtnText.textContent = 'Classifying...';

        const invertMode = invertModeSelect.value;

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: currentImageBase64,
                    invert_mode: invertMode
                })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Server returned an error during prediction.');
            }

            renderPredictionResults(data);

        } catch (err) {
            console.error('Prediction error:', err);
            showError(err.message || 'Failed to predict image. Make sure server is running.');
        } finally {
            loadingState.style.display = 'none';
            predictBtn.disabled = false;
            predictBtnText.textContent = 'Predict Image Category';
        }
    });

    function renderPredictionResults(data) {
        // Top prediction & confidence
        resCategory.textContent = data.prediction;
        resConfidence.textContent = `${data.confidence}%`;
        resDescription.textContent = data.description || '';

        // Dynamic badge color based on confidence
        if (data.confidence > 80) {
            confBadgeCircle.style.borderColor = '#10b981';
            confBadgeCircle.style.boxShadow = '0 0 16px rgba(16, 185, 129, 0.4)';
        } else if (data.confidence > 50) {
            confBadgeCircle.style.borderColor = '#f59e0b';
            confBadgeCircle.style.boxShadow = '0 0 16px rgba(245, 158, 11, 0.4)';
        } else {
            confBadgeCircle.style.borderColor = '#ef4444';
            confBadgeCircle.style.boxShadow = '0 0 16px rgba(239, 68, 68, 0.4)';
        }

        // Preprocessed 28x28 image preview
        if (data.preprocessed_image) {
            tensorPreprocessedImg.src = data.preprocessed_image;
        }
        if (data.was_inverted) {
            inversionStatusText.textContent = '✓ Light background detected: pixel intensity was inverted so foreground clothing is bright (matches Fashion-MNIST distribution).';
        } else {
            inversionStatusText.textContent = '✓ Dark background preserved: pixel distribution is directly aligned with Fashion-MNIST input tensors.';
        }

        // Probability breakdown bars
        probBarsList.innerHTML = '';
        if (Array.isArray(data.probabilities)) {
            data.probabilities.forEach((item, index) => {
                const isTop = index === 0;
                const row = document.createElement('div');
                row.className = 'prob-row';
                row.innerHTML = `
                    <span class="prob-name" title="${item.category}">${item.category}</span>
                    <div class="prob-track">
                        <div class="prob-fill ${isTop ? 'top-rank' : ''}" style="width: 0%"></div>
                    </div>
                    <span class="prob-num ${isTop ? 'top-rank' : ''}">${item.probability}%</span>
                `;
                probBarsList.appendChild(row);

                // Animate bar width smoothly
                setTimeout(() => {
                    const fill = row.querySelector('.prob-fill');
                    if (fill) fill.style.width = `${Math.max(item.probability, 1)}%`;
                }, 50 + index * 30);
            });
        }

        resultsContent.style.display = 'block';

        // Scroll into view if needed
        resultsContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // -------------------------------------------------------------
    // Metrics Fetch & Dynamic Table Update
    // -------------------------------------------------------------
    async function loadMetrics() {
        try {
            const resp = await fetch('/api/metrics');
            if (resp.ok) {
                const metrics = await resp.json();
                if (metrics.test_accuracy) {
                    const testAccEl = document.getElementById('metric-test-acc');
                    const valAccEl = document.getElementById('metric-val-acc');
                    const testLossEl = document.getElementById('metric-test-loss');
                    const epochsEl = document.getElementById('metric-epochs');

                    if (testAccEl) testAccEl.textContent = `${metrics.test_accuracy}%`;
                    if (valAccEl) valAccEl.textContent = `${metrics.val_accuracy}%`;
                    if (testLossEl) testLossEl.textContent = metrics.test_loss;
                    if (epochsEl) epochsEl.textContent = metrics.epochs || 12;

                    // Update hero stat as well
                    const heroStat = document.querySelector('.hero-stats .stat-number');
                    if (heroStat) heroStat.textContent = `${metrics.test_accuracy}%`;

                    // Populate detailed table
                    if (metrics.per_class_metrics) {
                        const tbody = document.getElementById('report-table-body');
                        if (tbody) {
                            tbody.innerHTML = '';
                            Object.entries(metrics.per_class_metrics).forEach(([clsName, m], idx) => {
                                const tr = document.createElement('tr');
                                tr.innerHTML = `
                                    <td>${idx}</td>
                                    <td><strong>${clsName}</strong></td>
                                    <td>${m.precision}%</td>
                                    <td>${m.recall}%</td>
                                    <td>${m.f1_score}%</td>
                                    <td>${m.support.toLocaleString()}</td>
                                `;
                                tbody.appendChild(tr);
                            });
                        }
                    }
                }
            }
        } catch (e) {
            console.log('Metrics loading error:', e);
        }
    }

    // -------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------
    function showError(msg) {
        errorMessage.textContent = msg;
        errorAlert.style.display = 'flex';
    }

    function hideError() {
        errorAlert.style.display = 'none';
        errorMessage.textContent = '';
    }

    // Initialize data
    loadSamples();
    loadMetrics();
});
