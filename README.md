# Bacterial Strain Identification System

## Project Overview
This repository contains the implementation of a deep learning model designed to identify different strains of bacteria present in petri dish images. Our approach combines YOLOv8 for instance segmentation with FastSAM for mask generation, creating a cost-efficient and computationally inexpensive solution for bacterial identification in urinary tract infections (UTIs).

Accurate identification of microbial strains is crucial for effective UTI treatment and prevention of severe complications like kidney damage. This solution is particularly valuable for rural medical institutions with limited access to advanced diagnostic equipment.

## Research Context
UTIs affect approximately 150 million people worldwide each year, with women, children, and the elderly being most vulnerable. If left untreated, UTIs can lead to serious health complications including permanent kidney damage and septic shock.

An estimated 70% of medical equipment designed in developed countries is unsuitable for developing regions due to infrastructure limitations. Our model offers a mobile-friendly alternative that requires only a camera and basic computing resources.

## Data
The dataset consists of nine common microbial strains cultured on Tryptic soy agar (TSA) and incubated at 37°C. Images were collected at four time intervals (18, 24, 48, and 72 hours), with multiple photos taken at each interval to enhance the model's generalization capabilities.

### Bacterial Strains
* Candida albicans
* Enterococcus faecalis
* Escherichia coli
* Klebsiella pneumoniae
* Pseudomonas aeruginosa
* Staphylococcus aureus
* Staphylococcus epidermidis
* Staphylococcus saprophyticus
* Streptococcus agalactiae

The dataset underwent rigorous preparation, including the removal of images with minimal growth (18-hour mark) or overgrowth (72-hour mark), as well as manual cropping to eliminate background objects.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/theCarterDavis/bacterial-strain-identification.git
   cd bacterial-strain-identification
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```


3. Download the pre-trained model weights:
   ```
   # The model weights file (best.pt) should be placed in the root directory
   # You can download it from our releases page or use the following command:
   wget https://github.com/theCarterDavis/bacterial-strain-identification/releases/download/v1.0/best.pt
   ```

## Quick Start Guide

### Running the Web Application
1. Navigate to the project directory:
   ```
   cd bacterial-strain-identification
   ```

2. Start the Flask application:
   ```
   python app.py
   ```

3. Open your web browser and go to `http://localhost:5000`

4. Upload an image of a bacterial culture in a petri dish and view the results

### Using Sample Images
The repository includes sample images in the `sample_images/` directory that you can use to test the application.

## Directory Structure
```
bacterial-strain-identification/
├── app.py                  # Flask web application
├── best.pt                 # Pre-trained model weights
├── requirements.txt        # Project dependencies
├── templates/              # HTML templates for the web app
│   ├── index.html          # Upload page
│   └── result.html         # Results display page
├── sample_images/          # Sample bacterial culture images
├── data/                   # Dataset used for training
├── notebooks/              # Jupyter notebooks for data preparation and analysis
├── scripts/                # Utility scripts for data processing
├── models/                 # Model architecture definitions
├── research_paper.pdf      # Research paper describing the methodology
└── README.md               # Project documentation
```

## Methodology
### Data Preparation
1. **Image Collection**: Bacterial strains were cultivated individually in petri dishes under sterile conditions and photographed at multiple time intervals.
2. **Preprocessing**: Images were manually cropped to remove background elements, and blurry images were excluded.
3. **Automated Mask Generation**: FastSAM was used to create segmentation masks for each bacterial colony.
4. **Data Filtering**: Masks with more than 300 data points were considered too large and filtered out. Images without valid masks were removed from the dataset.

### Model Architecture
Our implementation uses YOLOv8's segmentation variant, which incorporates:
- A Feature Pyramid Network (FPN) for multi-scale feature extraction
- A Path Aggregation Network (PAN) that combines features using skip connections
- Instance segmentation capabilities for precise colony identification

## Web Application Interface
The project includes a Flask-based web application that allows users to upload images of bacterial cultures for analysis:

### Features
- Simple upload interface for petri dish images
- Real-time bacterial strain identification
- Visual results showing original and segmented images
- Confidence levels for each identified bacterial strain
- Color-coded visualization of different bacterial strains

### Technical Implementation
- **Backend**: Python Flask server
- **Model Integration**: Direct integration with the YOLOv8 model
- **Visualization**: OpenCV-based segmentation with color overlays
- **User Interface**: Clean HTML/CSS interface optimized for both desktop and mobile use

### Usage
1. Start the Flask application:
   ```
   python app.py
   ```
2. Access the application through a web browser at `http://localhost:5000`
3. Upload an image of a petri dish with bacterial cultures
4. View the results showing identified bacterial strains with confidence scores

## Results
The model achieved the following performance metrics:
- Mean Average Precision (mAP50): 0.791
- Average Precision: 0.756 (σ = 0.111)
- Average Recall: 0.71 (σ = 0.089)

The confusion matrix demonstrated high accuracy across all bacterial strains, with correct identification rates of 87% or higher. The most accurately identified strain was Staphylococcus aureus, while Enterococcus faecalis proved more challenging due to limited valid data.

## Deployment Considerations
- The application can be deployed on low-cost hardware including Raspberry Pi devices
- For field use, the model can be optimized further using quantization techniques
- The system requires minimal bandwidth for operation once the model is loaded
- Docker containers are available for easy deployment in various environments

## Future Improvements
- Integration with microscope cameras for direct analysis
- Mobile application development for Android and iOS
- Expansion of the dataset to include additional bacterial strains
- Implementation of time-series analysis to track bacterial growth patterns
- Cloud-based version for remote access and centralized data collection

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - **Problem**: `RuntimeError: No such file or directory: 'best.pt'`
   - **Solution**: Ensure you've downloaded the model weights file and placed it in the root directory.

2. **CUDA Out of Memory**
   - **Problem**: `CUDA out of memory` error when processing large images
   - **Solution**: Reduce the image size before uploading or adjust the batch size in the app.py file.

3. **Flask App Not Starting**
   - **Problem**: `Address already in use` error when starting Flask
   - **Solution**: Change the port number in app.py or terminate the process using the current port.

4. **Image Format Issues**
   - **Problem**: `ValueError: cannot identify image file`
   - **Solution**: Ensure you're uploading supported image formats (PNG, JPG, JPEG, GIF).

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
We welcome contributions to improve the project! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

Please ensure your code follows our coding standards and includes appropriate tests.

## Citation
If you use this work in your research, please cite:

```
@article{davis2025identification,
  title={Identification of Microbial Strains Using an Instance Segmentation Model},
  author={Davis, Carter and Lange, Alexey and Blaszczyk, Sadie and Heldt, Heidi and Vander Schaaf, Nicole and Manjarrés, José},
  journal={},
  year={2025}
}
```

## Acknowledgments
* Departments of Mathematical, Information and Computer Sciences, Physics and Engineering at Point Loma Nazarene University
* Department of Biological Sciences at Olivet Nazarene University

## Contact Information
For questions, collaboration opportunities, or issues regarding this project, please contact:
- Carter Davis - cdavis0022@pointloma.edu

## Research Paper
The full research paper describing this project's methodology, experimental setup, and results can be found in the root directory as `research_paper.pdf`.

## References

Yang, X., Chen, H., Zheng, Y., Qu, S., Wang, H., & Yi, F. (2022). Disease burden and long-term trends of urinary tract infections: A worldwide report. Frontiers in Public Health, 10:888205.

Flores-Mireles, A. L., Walker, J. N., Caparon, M., & Hultgren, S. J. (2015). Urinary tract infections: epidemiology, mechanisms of infection and treatment options. Nature Reviews Microbiology, 13:269-284.

Vasan, A., & Friend, J. (2020). Medical devices for low- and middle-income countries: A review and directions for development. Journal of Medical Devices, 14:010803.

Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.

Sohan, M., Ram, T. S., & Reddy, C. V. R. (2024). A review on YOLOv8 and its advancements. In Jacob, I. J., Piramuthu, S., & Falkowski-Gilski, P. (Eds.), Data Intelligence and Cognitive Informatics (pp. 529-545). Springer Nature Singapore.

Zhao, X., Ding, W., An, Y., Du, Y., Yu, T., Li, M., Tang, M., & Wang, J. (2023). Fast segment anything. arXiv preprint.

Harley, A. W., Derpanis, K. G., & Kokkinos, I. (2017). Segmentation-aware convolutional networks using local attention masks. arXiv preprint arXiv:1708.04607.

Andreini, P., Bonechi, S., Bianchini, M., Mecocci, A., & Di Massa, V. (2015). Automatic image analysis and classification for urinary bacteria infection screening. In Murino, V. & Puppo, E. (Eds.), Image Analysis and Processing — ICIAP 2015 (pp. 635-646). Springer International Publishing.

Chauhan, V. K., Dahiya, K., & Sharma, A. (2019). Problem formulations and solvers in linear SVM: a review. Artificial Intelligence Review, 52(2):803-855.

Wang, H., Koydemir, H. C., Qiu, Y., Bai, B., Zhang, Y., Jin, Y., Tok, S., Yilmaz, E. C., Gumustekin, E., Rivenson, Y., & Ozcan, A. (2020). Early detection and classification of live bacteria using time-lapse coherent imaging and deep learning. Light: Science & Applications, 9(1):118.

Breakwell, D., Woolverton, C., MacDonald, B., Smith, K., & Robison, R. (2007). Colony morphology protocol. Technical report, American Society for Biology.

Fan, J., Lee, J., Jung, I., & Lee, Y. (2021). Improvement of object detection based on faster R-CNN and YOLO. In 2021 36th International Technical Conference on Circuits/Systems, Computers and Communications (ITC-CSCC) (pp. 1-4).

Gündüz, M. Ş., & Işık, G. (2023). A new YOLO-based method for real-time crowd detection from video and performance analysis of YOLO models. Journal of Real-Time Image Processing, 20(1):5.

Oymak, S., Li, M., & Soltanolkotabi, M. (2021). Generalization guarantees for neural architecture search with train-validation split. In International Conference on Machine Learning (pp. 8291-8301).

Ferro, M. V., Mosquera, Y. D., Pena, F. J. R., & Bilbao, V. M. D. (2023). Early stopping by correlating online indicators in neural networks. Neural Networks, 159:109-124.

Davis, J., & Goadrich, M. (2006). The relationship between precision-recall and ROC curves. In Proceedings of the 23rd International Conference on Machine Learning (pp. 233-240).

Novakovic, J., Veljovic, A., Ilic, S., Papic, Ž. M., & Milica, T. (2017). Evaluation of classification models in machine learning. Theory and Applications of Mathematics & Computer Science, 7:39-46.
