# Face Glow AI

**Face Glow AI** is a Single Page Application (SPA) built using **Django (backend)** and **React (frontend)** that enhances facial images using advanced AI techniques.

## 🔍 Features

- Upload an image
- Detect and crop faces
- Denoise facial regions
- Reintegrate processed faces into the original image
- Enhance the entire image using **GFPGAN**
- Sharpen the image
- Improve colors and brightness
- Preview and **download the enhanced image**
- 100% Free – **No signup required**

## 🧠 Tech Stack

- **Backend**: Django + GFPGAN (AI model)
- **Frontend**: React (SPA)
- **Image Processing**: OpenCV, GFPGAN, and enhancement filters

## 🖼️ How It Works

1. User uploads an image.
2. Face is detected and cropped.
3. Cropped face is denoised.
4. Processed face is reintegrated into the original image.
5. Whole image is enhanced using GFPGAN.
6. Further sharpening, color correction, and brightness improvement is applied.
7. Final enhanced image is returned.
8. User can view and download the result.

## 🖥️ Intefaces


### Main Interface

![Preview](.\faceglow\static\faceglow\Images\Mainpage.png)


### Image Processing Interface

![Preview](.\faceglow\static\faceglow\Images\Processing.png)


### Final Output Interface

![Preview](.\faceglow\static\faceglow\Images\Show_Result.png)