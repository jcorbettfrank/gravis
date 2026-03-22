// WebGPU feature detection for live demos
// If WebGPU is unavailable, hide iframes and show fallback screenshots.
document.addEventListener('DOMContentLoaded', function() {
    if (!navigator.gpu) {
        document.querySelectorAll('.live-demo').forEach(function(demo) {
            var iframe = demo.querySelector('iframe');
            var fallback = demo.querySelector('.demo-fallback');
            if (iframe) iframe.style.display = 'none';
            if (fallback) fallback.style.display = 'block';
        });
    }
});
