# FloorPlanTo3D: The Next Dimension in Architectural Visualization

## Introduction

FloorPlanTo3D introduces an innovative approach to transform 2D floor plan images into customizable 3D models. The process begins with users selecting a floor plan image, which is then sent to a deep learning model for analysis. Subsequently, users have the ability to customize the generated 3D scene according to their preferences.
![2D Floor Plan to 3D Model Conversion - Example a](./images/example1.png) ![2D Floor Plan to 3D Model Conversion - Example b](./images/example2.png) ![2D Floor Plan with Hand Drawn Parts to 3D Model Conversion](./images/handDrawn.png)

FloorPlanTo3D showcases the robust capability to detect and interpret floor plans with different drawing styles, including those with hand-drawn elements.

## Project Components

FloorPlanTo3D is structured into two main parts:

1.  **Mask R-CNN Model Wrapped by REST API**: A sophisticated deep learning model that analyzes and interprets 2D floor plan images. **Now supports multiple output formats for different frontend technologies.**

2.  **Frontend Applications**: 
    - **Unity Application**: The original dynamic application that constructs 3D scenes
    - **Web Frontend**: New web-based interface with modern UI
    - **Adaptable Output**: API can be consumed by any frontend (React, Vue, Three.js, etc.)

## Installation

To set up and run the project, follow these steps:

1.  **Clone this repo**:

```
git clone https://github.com/fadyazizz/FloorPlanTo3D-API
cd FloorPlanTo3D-API

```

2.  **Create and activate a new conda environment**:

```

conda create --name imageTo3D python=3.6.13

conda activate imageTo3D

```

3.  **Install dependencies**:

```


pip install -r requirements.txt

```

4.  **Download the deep learning model weights** from the following link and insert them into the `weights` folder:

[Deep Learning Model Weights](https://drive.google.com/file/d/14fDV0b_sKDg0_DkQBTyO1UaT6mHrW9es/view?usp=sharing)

5.  **Start the server**:

```

python application.py

```

These steps will prepare your environment for using the API. The API now supports multiple output formats and can be integrated with various frontend technologies:

- **Unity Client**: [Our Unity Client](https://github.com/fadyazizz/FloorPlanTo3D-unityClient) (Unity engine installation required)
- **Web Frontend**: See `web_example/` directory for a complete web interface
- **Custom Integration**: Use any of the available output formats for your preferred frontend

## Customization Features, download from this link [Our Unity Client](https://github.com/fadyazizz/FloorPlanTo3D-unityClient)

Users are afforded a wide range of customization options for their 3D models, including but not limited to:

- Snap an image and send to the api to be analyzed
- Modifying the scale to ensure the model matches real-world dimensions.

  <div style="display: flex; align-items: center; justify-content: space-around;">
    <img src="./images/scale2.png" width="500" height="250 alt="Alt text for first image">
    <img src="./images/scale1.png" width="500" height="250 alt="Alt text for second image">
</div>

- Changing the colors and textures of walls.
 <div style="display: flex; align-items: center; justify-content: space-around;">
    <img src="./images/wall1.png" width="500" height="250 alt="Alt text for first image">
    <img src="./images/wall2.png" width="500" height="250 alt="Alt text for second image">
</div>

- Adding furniture and selecting different styles for doors and windows.
<div style="display: flex; align-items: center; justify-content: space-around;">
   <img src="./images/furniture.png" width="500" height="250 alt="Alt text for first image">

</div>

- Have a virtual tour inside the 3D generated floor plan.

## Model Used

FloorPlanTo3D employs the Mask R-CNN model, renowned for its accuracy in object detection and instance segmentation. Our implementation is based on the Matterport version, which is specifically adapted to analyze floor plans effectively. For further details on the Mask R-CNN model, visit the [Mask R-CNN GitHub Repository](https://github.com/matterport/Mask_RCNN).

## Model Training

The model training process involved the following key steps:

- **Data set management**: Utilized a split of 80% training and 20% testing from the Cubicasa5K dataset, which contains 5000 floor plans with diverse drawing styles and is available at [Cubicasa5K Dataset](https://zenodo.org/record/2613548).

- **Model configuration**: Employed Resnet101 as the backbone, with transfer learning from the MS COCO dataset to enhance training efficiency.

- **Training**: Conducted over 15 epochs with a batch size of 1, completed in approximately 40 hours, to detect three classes of objects: walls, windows, and doors.

For an in-depth exploration of the project, refer to the bachelor's thesis available at: [Bachelor Thesis Link](https://drive.google.com/file/d/11xyyv_jUtbEp0WM-ymfffnzX45ryDV0X/view?usp=sharing).

## 🔧 API Usage & Output Formats

The API now supports multiple output formats to accommodate different frontend technologies:

### Available Endpoints

- `POST /` or `POST /predict` - Main prediction endpoint
- `GET /formats` - Get information about available output formats

### Output Formats

#### 1. Unity Format (Default)
```bash
curl -X POST -F "image=@floorplan.jpg" "http://localhost:5000/?format=unity"
```

#### 2. Web Format (Optimized for web applications)
```bash
curl -X POST -F "image=@floorplan.jpg" "http://localhost:5000/?format=web"
```

#### 3. Three.js Format (3D scene data)
```bash
curl -X POST -F "image=@floorplan.jpg" "http://localhost:5000/?format=threejs"
```

### Integration Examples

#### JavaScript/Web
```javascript
const analyzeFloorPlan = async (imageFile, format = 'web') => {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch(`http://localhost:5000/predict?format=${format}`, {
    method: 'POST',
    body: formData
  });
  
  return await response.json();
};
```

#### Python
```python
import requests

def analyze_floorplan(image_path, format='web'):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f'http://localhost:5000/predict?format={format}', files=files)
    return response.json()
```

#### cURL
```bash
# Analyze with web format
curl -X POST -F "image=@your_floorplan.jpg" "http://localhost:5000/predict?format=web"

# Get available formats
curl -X GET "http://localhost:5000/formats"
```

### Web Example
Run the included web example:
```bash
cd web_example
python -m http.server 8080
# Open http://localhost:8080 in your browser
```

## Author

Fady Aziz Ibrahim

Email: fady.aziz.ibrahim@gmail.com

LinkedIn: [Fady Aziz](https://www.linkedin.com/in/fady-aziz-b40687163/)
