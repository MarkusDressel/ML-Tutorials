import os
import gradio as gr
import numpy as np
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
import PIL


processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForTokenClassification.from_pretrained("MarkusDressel/cord")
id2label = model.config.id2label

label_ints = np.random.randint(0,len(PIL.ImageColor.colormap.items()),30)

label_color_pil = [k for k,_ in PIL.ImageColor.colormap.items()]

label_color = [label_color_pil[i] for i in label_ints]
label2color = {}
for k,v in id2label.items():
    label2color[v[2:]]=label_color[k]


def unnormalize_box(bbox, width, height):
     return [
         width * (bbox[0] / 1000),
         height * (bbox[1] / 1000),
         width * (bbox[2] / 1000),
         height * (bbox[3] / 1000),
     ]
def iob_to_label(label):
    label = label[2:]
    if not label:
        return 'other'
    return label


def process_image(image):
    width, height = image.size
    # encode
    encoding = processor(image, truncation=True, return_offsets_mapping=True, return_tensors="pt")
    offset_mapping = encoding.pop('offset_mapping')
    # forward pass
    outputs = model(**encoding)
    # get predictions
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    token_boxes = encoding.bbox.squeeze().tolist()
    # only keep non-subword predictions
    is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
    true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
    true_boxes = [unnormalize_box(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(true_predictions, true_boxes):
        predicted_label = iob_to_label(prediction).lower()
        draw.rectangle(box, outline=label2color[predicted_label], width=5)
        draw.text((box[0]+10, box[1]-10), text=predicted_label, fill=label2color[predicted_label], font=font)
    
    return image
title = "Cord demo: LayoutLMv2"
description = "Demo for Microsoft's LayoutLMv2.This particular model is fine-tuned on CORD, a dataset of manually annotated receipts. It annotates the words appearing in the image in up to 30 classes. To use it, simply upload an image or use the example image below and click 'Submit'. Results will show up in a few seconds. If you want to make the output bigger, right-click on it and select 'Open image in new tab'."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2012.14740' target='_blank'>LayoutLMv2: Multi-modal Pre-training for Visually-Rich Document Understanding</a> | <a href='https://github.com/microsoft/unilm' target='_blank'>Github Repo</a></p>"
examples =[['receipt_00189.png']]
css = ".output_image, .input_image {height: 40rem !important; width: 100% !important;}"
#css = "@media screen and (max-width: 600px) { .output_image, .input_image {height:20rem !important; width: 100% !important;} }"
# css = ".output_image, .input_image {height: 600px !important}"
iface = gr.Interface(fn=process_image, 
                     inputs=gr.inputs.Image(type="pil"), 
                     outputs=gr.outputs.Image(type="pil", label="annotated image"),
                     title=title,
                     description=description,
                     article=article,
                     examples=examples,
                     css=css,
                     enable_queue=True)
iface.launch(debug=True)