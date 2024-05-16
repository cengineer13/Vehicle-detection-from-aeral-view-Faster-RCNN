
# Vehicle object detection Faster RCNN

## Project Overview
This project is purposed to detect vehicles from aerial view of camera.  
## Installation

To set up the project environment:
#### 1-method

```bash
git clone https://github.com/cengineer13/Vehicle-detection-from-aeral-view-Faster-RCNN.git
cd Vehicle-detection-from-aeral-view-Faster-RCNN
pip install -r requirements.txt
```

#### 2-method

Create conda environment using two methods:

a) Create a virtual environment using yml file:

```python
conda env create -f environment.yml
```

Then activate the environment using the following command:
```python
conda activate ds
```

b) Create a virtual environment using txt file:

- Create a virtual environment:

```python
conda create -n ds python=3.10
```

- Activate the environment using the following command:

```python
conda activate ds
```

- Install libraries from the text file:

```python
pip install -r requirements.txt
```

## Dataset

The dataset contains 259 brain CT images classified into aneurysm, tumor, and cancer classes.
* Download dataset from the [link](https://www.kaggle.com/datasets/killa92/brain-ct-tumor-classification-dataset).

<h4 align="center"> Training dataset examples</h4>

![Train Random Examples](data/plots/1-train_random_examples.png)

<h4 align="center"> Validation dataset examples</h4>

![Validation Random Examples](data/plots/1-val_random_examples.png)

<h4 align="center"> Test dataset examples</h4>

![Test Random Examples](data/plots/1-test_random_examples.png)

These images are random visualizations from the training, validation, and test sets.

## Model Performance

The model's training and validation metrics are visualized below:
![Training and Validation Accuracy](data/plots/2-Training%20and%20Validation%20accuracy%20metrics.png)
![Training and Validation Loss](data/plots/2-Training%20and%20Validation%20loss%20metrics.png)

The curves represent the model's accuracy and loss over epochs, indicating the learning process and convergence.

## Inference

Here's how the model performs on the test set:

![Inference Results](/data/plots/3-Inference_result_examples.png)

Each row shows the ground truth and predicted label for brain CT images.

## GradCAM Visualizations

The model's attention is visualized using GradCAM, which helps understand which regions of the images influenced the predictions:

![GradCAM Results](/data/plots/4-GradCam_results_examples.png)

Each image highlights the areas most influential for the model's predictions.

### Arguments for training 
* Train the model using the following arguments:

![image](data/assets/main_arguments.png)

```python

python main.py  --batch_size 32 --epochs 15

```
* Inference demo process with trained model using the following arguments:

![image](data/assets/demo_arguments.png)

```python

python demo.py--model_name "brain"  --test_img "path_to_test_image" 

```


## Testing on New Data &  Interactive Web Demo

A Streamlit web app (`demo.py`) allows users to upload an image and receive model predictions in a user-friendly interface.

```bash
streamlit run demo.py
```
Result: 
![Streamlit demo](/data/assets/demo.png)

To evaluate the model on new, I used this unseen images:

![New Test Image Cancer 2](/data/test_images/cancer2.jpg)
![New Test Image Tumor](/data/test_images/tumor_test.jpg)

These are examples of new data images passed through the model.


## Contributing

We welcome contributions from the community. Please read the contributing guidelines first before making a pull request.

## License

This project is licensed under the MIT License - see [LICENSE.md](LICENSE.md) for details.

## Contact

Mr Murodil  - murodild@gmail.com
LinkedIn - https://www.linkedin.com/in/mr-murodil
Project Link: [https://github.com/yourusername/brain-ct-classification](https://github.com/yourusername/brain-ct-classification)
