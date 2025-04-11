document.addEventListener('DOMContentLoaded', () => {
    const uploadBox = document.getElementById('uploadBox');
    const imageInput = document.getElementById('imageInput');
    const uploadContainer = document.getElementById('uploadContainer');
    const progressContainer = document.getElementById('progressContainer');
    const resultContainer = document.getElementById('resultContainer');
    const originalImage = document.getElementById('originalImage');
    const enhancedImage = document.getElementById('enhancedImage');
    const downloadBtn = document.getElementById('downloadBtn');
    const newImageBtn = document.getElementById('newImageBtn');

    // For demo purposes - using the same image
    const demoImage = 'https://images.unsplash.com/photo-1581833971358-2c8b550f87b3?auto=format&fit=crop&q=80&w=1000';

    // Upload handling
    uploadBox.addEventListener('click', () => imageInput.click());

    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = '#4facfe';
        uploadBox.style.background = 'rgba(79, 172, 254, 0.05)';
    });

    uploadBox.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        uploadBox.style.background = 'transparent';
    });

    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            handleImage(file);
        }
    });

    imageInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleImage(file);
        }
    });

    function handleImage(file) {
        // Show progress
        uploadContainer.style.display = 'none';
        progressContainer.style.display = 'block';
        const formData = new FormData();
        formData.append('image', file);
        const csrfToken = document.querySelector('meta[name="csrf-token"]').getAttribute('content')

        showAnnimation()
        fetch('enhance-image/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': csrfToken
            },
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                console.log(data)
                // Get the base64 image string from the response
                const base64Image = data.image;
                // Find your <img> tag and set the src to the base64 string
                enhancedImage.src = base64Image;
                showResult(file, base64Image)
            })
    }

    function showAnnimation() {
        // Demo progress animation
        const steps = document.querySelectorAll('.step');
        let currentStep = 0;

        function nextStep() {
            if (currentStep > 0) {
                steps[currentStep - 1].classList.remove('active');
            }
            steps[currentStep].classList.add('active');
            currentStep++;

            if (currentStep < steps.length) {
                let nextTime = 0;
                if (currentStep === 1) {
                    nextTime = 3000;
                }
                else if (currentStep === 4) {
                    nextTime = 12000; // Step 1, 3 seconds
                }
                else {
                    nextTime = 2000; // Steps 2-4, 2 seconds
                }

                setTimeout(nextStep, nextTime); // Call nextStep after the determined time
            }
        }
        nextStep(); // Start the animation
    }


    function showResult(original_image, enhanced_image) {
        progressContainer.style.display = 'none';
        resultContainer.style.display = 'block';

        // For demo - using the same image
        originalImage.src = URL.createObjectURL(original_image);
        enhancedImage.src = enhanced_image;

        downloadBtn.addEventListener('click', () => {
            // In real implementation, this would download the enhanced image
            downloadImage(enhanced_image)
        });

        initializeSlider();
    }

    // Image comparison slider
    function initializeSlider() {
        const slider = document.querySelector('.comparison-slider');
        const handle = document.querySelector('.slider-handle');
        let isResizing = false;

        handle.addEventListener('mousedown', (e) => {
            isResizing = true;
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', () => {
                isResizing = false;
                document.removeEventListener('mousemove', onMouseMove);
            });
        });

        handle.addEventListener('touchstart', (e) => {
            isResizing = true;
            document.addEventListener('touchmove', onTouchMove);
            document.addEventListener('touchend', () => {
                isResizing = false;
                document.removeEventListener('touchmove', onTouchMove);
            });
        });

        function onMouseMove(e) {
            if (!isResizing) return;
            const sliderRect = slider.getBoundingClientRect();
            const x = Math.max(0, Math.min(e.clientX - sliderRect.left, sliderRect.width));
            const percentage = (x / sliderRect.width) * 100;
            updateSliderPosition(percentage);
        }

        function onTouchMove(e) {
            if (!isResizing) return;
            const sliderRect = slider.getBoundingClientRect();
            const x = Math.max(0, Math.min(e.touches[0].clientX - sliderRect.left, sliderRect.width));
            const percentage = (x / sliderRect.width) * 100;
            updateSliderPosition(percentage);
        }

        function updateSliderPosition(percentage) {
            handle.style.left = `${percentage}%`;
            document.querySelector('.slider-line').style.left = `${percentage}%`;
            enhancedImage.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
        }
    }

    // New image button
    newImageBtn.addEventListener('click', () => {
        resultContainer.style.display = 'none';
        uploadContainer.style.display = 'block';
        const steps = document.querySelectorAll('.step');
        steps.forEach(step => step.classList.remove('active'));
    });

    function downloadImage(base64Image) {
        if (!base64Image) {
            console.log("No Image");
            return;
        }

        const [header, data] = base64Image.split(',');
        const mime = header.match(/:(.*?);/)[1];
        const byteCharacters = atob(data);
        const byteArrays = [];

        for (let i = 0; i < byteCharacters.length; i += 512) {
            const slice = byteCharacters.slice(i, i + 512);
            const byteNumbers = new Array(slice.length);
            for (let j = 0; j < slice.length; j++) {
                byteNumbers[j] = slice.charCodeAt(j);
            }
            byteArrays.push(new Uint8Array(byteNumbers));
        }

        const blob = new Blob(byteArrays, { type: mime });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "enhanced_image.jpg";
        link.click();
        URL.revokeObjectURL(link.href);
    }
});