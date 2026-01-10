let distanceChart;
let currentMode = 'auto';

function initChart() {
    const ctx = document.getElementById('distanceChart').getContext('2d');
    distanceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Finger Distance (px)',
                data: [],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 220,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        callback: function(value) {
                            return value + 'px';
                        }
                    }
                },
                x: {
                    display: false
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    enabled: true,
                    callbacks: {
                        label: function(context) {
                            return context.parsed.y.toFixed(1) + ' px';
                        }
                    }
                }
            }
        }
    });
}

function updateCircularProgress(elementId, value, maxValue) {
    const circle = document.getElementById(elementId);
    const circumference = 2 * Math.PI * 45;
    const percentage = (value / maxValue) * 100;
    const offset = circumference - (percentage / 100) * circumference;
    circle.style.strokeDashoffset = offset;
}

function updateMetrics() {
    fetch('/metrics')
        .then(response => response.json())
        .then(data => {
            document.getElementById('volumeValue').textContent = data.volume.toFixed(1) + '%';
            document.getElementById('volumeLevel').textContent = data.volume_level;
            document.getElementById('distanceValue').textContent = data.distance.toFixed(1) + 'px';
            document.getElementById('fpsValue').textContent = data.fps.toFixed(1);
            document.getElementById('responseValue').textContent = data.response_time.toFixed(1) + 'ms';

            updateCircularProgress('volumeCircle', data.volume, 100);
            updateCircularProgress('distanceCircle', data.distance, 200);
            updateCircularProgress('fpsCircle', data.fps, 10);
            updateCircularProgress('responseCircle', 100 - Math.min(data.response_time, 100), 100);

            document.getElementById('volumeFill').style.width = data.volume + '%';
            document.getElementById('volumePercentage').textContent = data.volume.toFixed(0) + '%';

            document.getElementById('gestureName').textContent = data.gesture;

            const lockStatus = document.getElementById('lockStatus');
            const lockIcon = lockStatus.querySelector('.lock-icon');
            const lockText = lockStatus.querySelector('.lock-text');

            if (data.locked) {
                lockIcon.textContent = '🔒';
                lockText.textContent = 'Locked';
                lockStatus.style.background = 'rgba(255, 0, 0, 0.3)';
            } else {
                lockIcon.textContent = '🔓';
                lockText.textContent = 'Unlocked';
                lockStatus.style.background = 'rgba(255, 255, 255, 0.2)';
            }

            if (data.calibration_mode === 'auto') {
                document.getElementById('autoMin').textContent = data.auto_min.toFixed(1) + 'px';
                document.getElementById('autoMax').textContent = data.auto_max.toFixed(1) + 'px';
            }

            if (distanceChart && data.distance_history) {
                const history = data.distance_history.slice(-50);
                distanceChart.data.labels = history.map((_, i) => i);
                distanceChart.data.datasets[0].data = history;
                distanceChart.update('none');
            }
        })
        .catch(error => {
            console.error('Error fetching metrics:', error);
        });
}

function changeCalibrationMode(mode) {
    currentMode = mode;

    const autoCalibration = document.getElementById('autoCalibration');
    const manualCalibration = document.getElementById('manualCalibration');

    if (mode === 'auto') {
        autoCalibration.style.display = 'flex';
        manualCalibration.style.display = 'none';

        fetch('/set_calibration', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mode: 'auto' })
        });
    } else {
        autoCalibration.style.display = 'none';
        manualCalibration.style.display = 'flex';
    }
}

function updateManualCalibration() {
    const minValue = document.getElementById('manualMinSlider').value;
    const maxValue = document.getElementById('manualMaxSlider').value;

    document.getElementById('manualMinValue').textContent = minValue;
    document.getElementById('manualMaxValue').textContent = maxValue;
}

function applyManualCalibration() {
    const minValue = parseInt(document.getElementById('manualMinSlider').value);
    const maxValue = parseInt(document.getElementById('manualMaxSlider').value);

    fetch('/set_calibration', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            mode: 'manual',
            min: minValue,
            max: maxValue
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Calibration settings applied successfully!');
        }
    })
    .catch(error => {
        console.error('Error applying calibration:', error);
    });
}

function logout() {
    window.location.href = '/logout';
}

document.addEventListener('DOMContentLoaded', function() {
    initChart();
    updateMetrics();
    setInterval(updateMetrics, 100);
});
