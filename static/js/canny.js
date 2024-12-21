// Image Data Arrays
let uint8_lena = lenna_256x256;
let uint16_lena = lenna_256x256.map(px => px * 256);
let float_lena = lenna_256x256.map(px => px / 256);

// Image dimensions
let W = 256;
let H = 256;

// Creating Image with uint8 data
let img = new T.Image('uint8', W, H);
img.setPixels(new Uint8Array(uint8_lena));

// Graphics Context for preview
let gpuEnv = gpu.getGraphicsContext("preview");
gpuDisplay(img.getRaster(), gpuEnv);

// Creating Image with uint8 data for edge detection
let img1 = new T.Image('uint8', W, H);
img1.setPixels(new Uint8Array(uint8_lena));
let gpuEnv1 = gpu.getGraphicsContext("preview1");
gpuEdgeCanny(50.0, 100.0)(img1.getRaster(), gpuEnv1);

// Creating Image with uint16 data for edge detection
let img2 = new T.Image('uint16', W, H);
img2.setPixels(new Uint16Array(uint16_lena));
let gpuEnv2 = gpu.getGraphicsContext("preview2");
gpuEdgeCanny(50.0, 100.0)(img2.getRaster(), gpuEnv2);

// Creating Image with float32 data for edge detection
let img3 = new T.Image('float32', W, H);
img3.setPixels(new Float32Array(float_lena));
let gpuEnv3 = gpu.getGraphicsContext("preview3");
gpuEdgeCanny(50.0, 100.0)(img3.getRaster(), gpuEnv3);

