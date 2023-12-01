import gradio as gr
import os
import torch
from timeit import default_timer as timer
from typing import Tuple, Dict
from model import effb2_model
from pathlib import Path

# Setup class names
class_names = ["pizza", "steak", "sushi"]

### 2. Model and transforms preparation ###

# Create EffNetB2 model
eff_model, effb2_transform = effb2_model()

# Load saved weights
eff_model.load_state_dict(torch.load(f="Effnet_b2_10_epochs.pth",
        map_location=torch.device(type="cpu")))

### 3. Predict function ###

# Create predict function
def predict(img) -> Tuple[Dict, float]:
    
    start_time = timer()
    
    # Transform the target image and add a batch dimension
    transformed_img = effb2_transform(img).unsqueeze(dim=0)
    
    # Put model into evaluation mode and turn on inference mode
    eff_model.eval()
    with torch.inference_mode():
        logits = eff_model(transformed_img)
        # Pass the transformed image through the model and turn the prediction logits into prediction probabilities
        pred_probs = torch.softmax(logits, dim=1)

    end_time = timer()
    pred_time = round(end_time - start_time, 5)

    # Create a prediction label and prediction probability dictionary for each prediction class (this is the required format for Gradio's output parameter)
    predictions_dict = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
       
    # Return the prediction dictionary and prediction time 
    return predictions_dict, pred_time

### 4. Gradio app ###

# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Pytorch Model Deployment"

# Create examples list from "examples/" directory
example_path  = Path('examples')
example_list = [["examples/" + example] for example in os.listdir(example_path)]


#Create the Gradio demo
demo = gr.Interface(fn=predict, # mapping function from input to output
                    inputs=gr.Image(type="pil"), # what are the inputs?
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), # what are the outputs?
                             gr.Number(label="Prediction time (s)")], # our fn has two outputs, therefore we have two outputs
                    # Create examples list from "examples/" directory
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch(debug=False, share=True)