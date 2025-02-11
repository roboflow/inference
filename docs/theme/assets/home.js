gsap.registerPlugin(ScrollTrigger);

function runHomeScript() {

    // Wait for DOM to be fully loaded
    setTimeout(() => {
        // Function to create a Rive instance
        function createRiveInstance(canvasId, src, stateMachines) {
            const canvas = document.getElementById(canvasId);
            if (canvas) {
                const riveInstance = new rive.Rive({
                    src: src,
                    canvas: canvas,
                    autoplay: true,
                    stateMachines: stateMachines,
                    artboard: 'Artboard',
                    onLoad: () => {
                        riveInstance.resizeDrawingSurfaceToCanvas();
                    },
                });

                // Function to handle resizing
                const resizeCanvas = () => {
                    // Set canvas dimensions to match its parent
                    canvas.width = canvas.clientWidth;
                    canvas.height = canvas.clientHeight;
                    riveInstance.resizeDrawingSurfaceToCanvas();
                };

                // Add event listener for window resize
                window.addEventListener('resize', resizeCanvas);
                resizeCanvas(); // Initial resize
            }
        }

        // Create Rive instances for different animations
        if (document.querySelector('#rive-block-ticker')) {
            createRiveInstance('rive-block-ticker', '../assets/rive/rive-block-ticker.riv', ["State Machine 1"]);
        }

        if (document.querySelector('#rive-model-chaining')) {
            ScrollTrigger.create({
                trigger: '#rive-model-chaining',
                start: "top bottom", // Adjust as needed
                onEnter: () => {
                    createRiveInstance('rive-model-chaining', '../assets/rive/rive-model-chaining2.riv', ["State Machine 1"]);
                },
            });
        }

        if (document.querySelector('#rive-extend-custom-code')) {
            ScrollTrigger.create({
                trigger: '#rive-extend-custom-code',
                start: "center bottom", // Adjust as needed
                onEnter: () => {
                    createRiveInstance('rive-extend-custom-code', '../assets/rive/rive-extend-custom-code.riv', ["State Machine 1"]);
                },
            });
        }

        if (document.querySelector('#tile-grid')) {
            ScrollTrigger.create({
                trigger: '#tile-grid',
                start: "top bottom",
                onEnter: () => {
                    gsap.fromTo('#tile-grid .tile',
                        { opacity: 0, y: 20 },
                        { opacity: 1, y: 0, duration: 0.3, stagger: 0.1 }
                    );
                },
            });
        }

        if (document.querySelector('#rive-ml-cl-barcode')) {
            ScrollTrigger.create({
                trigger: '#rive-ml-cl-barcode',
                start: "center bottom", // Adjust as needed
                onEnter: () => {
                    createRiveInstance('rive-ml-cl-barcode', '../assets/rive/rive-ml-cl-barcode.riv', ["State Machine 1"]);
                },
            });
        }

        if (document.querySelector('#notifications')) {
            ScrollTrigger.create({
                trigger: '#notifications',
                start: "top bottom", // Adjust as needed
                onEnter: () => {
                    gsap.fromTo('#notifications .notification',
                        { opacity: 0, y: 20 },
                        { opacity: 1, y: 0, duration: 0.5, stagger: 0.1 }
                    );
                },
            });
        }

        if (document.querySelector('#deployments')) {
            ScrollTrigger.create({
                trigger: '#deployments',
                start: "top bottom", // Adjust as needed
                onEnter: () => {
                    gsap.fromTo('#deployments .card-deployment', { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.5, stagger: 0.1 }
                    );
                },
            });
        }

        // Initialize Swiper
        if (document.querySelector('.swiper')) {
            new Swiper('.swiper', {
                slidesPerView: 1,
                spaceBetween: 16,
                breakpoints: {
                    1024: {
                        slidesPerView: 2,
                    }
                },
                pagination: {
                    el: '.swiper-pagination',
                    clickable: true,
                }
            });
        }
    }, 100);
}

document$.subscribe(runHomeScript);
