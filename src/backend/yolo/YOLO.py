# -*- coding: utf-8 -*-

import cv2
import numpy as np
import onnxruntime


class YOLO26Base:
    def initialize_model(self, path):
        providers = onnxruntime.get_available_providers()
        provider_options = []
        for provider in providers:
            if provider == "CoreMLExecutionProvider":
                provider_options.append({
                    "ModelFormat": "MLProgram",
                })
            else:
                provider_options.append({})
        self.session = onnxruntime.InferenceSession(
            path, providers=providers, provider_options=provider_options
        )
        self.get_input_details()
        self.get_output_details()

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    def prepare_input(self, image: np.ndarray):
        self.img_height, self.img_width = image.shape[:2]

        # Resize and convert BGR to RGB
        input_img = cv2.resize(image, (self.input_width, self.input_height))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # Scale pixel values to 0-1 and CHW format
        input_img = input_img.astype(np.float32) / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :]

        return input_tensor

    def inference(self, input_tensor):
        return self.session.run(self.output_names, {self.input_names[0]: input_tensor})


class YOLO26(YOLO26Base):
    """
    YOLO26 Object Detection (NMS-free / End-to-End)
    Output format: [x1, y1, x2, y2, score, class_id]
    """

    def __init__(self, path, conf_thres=0.25):
        self.conf_threshold = conf_thres
        self.initialize_model(path)

    def __call__(self, image: np.ndarray):
        return self.detect_objects(image)

    def detect_objects(self, image: np.ndarray):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)

        # YOLO26 End-to-End output is typically [1, 300, 6]
        # [x1, y1, x2, y2, score, class_id]
        return self.process_output(outputs)

    def process_output(self, output):
        predictions = np.squeeze(output[0])  # Shape: [300, 6]

        # Filter by confidence
        scores = predictions[:, 4]
        mask = scores > self.conf_threshold
        valid_predictions = predictions[mask]

        if len(valid_predictions) == 0:
            return np.array([]), np.array([]), np.array([])

        boxes = valid_predictions[:, :4]
        scores = valid_predictions[:, 4]
        class_ids = valid_predictions[:, 5].astype(int)

        # Rescale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        return boxes, scores, class_ids

    def rescale_boxes(self, boxes):
        # YOLO26 coordinates are usually absolute relative to the input_size (e.g. 0-640)
        # We scale them based on the ratio between original image and model input
        rescaled_boxes = np.zeros_like(boxes)
        rescaled_boxes[:, [0, 2]] = boxes[:, [0, 2]] * (
            self.img_width / self.input_width
        )
        rescaled_boxes[:, [1, 3]] = boxes[:, [1, 3]] * (
            self.img_height / self.input_height
        )
        return rescaled_boxes


class YOLO26Cls(YOLO26Base):
    """
    YOLO26 Image Classification
    """

    def __init__(self, path, conf_thres=0.7):
        self.conf_threshold = conf_thres
        self.initialize_model(path)

    def __call__(self, image: np.ndarray):
        return self.predict(image)

    def predict(self, image: np.ndarray):
        input_tensor = self.prepare_input(image)

        # Specific normalization for Classification (ImageNet stats)
        # Often required if exported from Ultralytics
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
        input_tensor = (input_tensor - mean) / std

        outputs = self.inference(input_tensor)
        return self.process_output(outputs)

    def process_output(self, output):
        logits = np.squeeze(output[0])

        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        # Get classes above threshold
        class_ids = np.where(probs > self.conf_threshold)[0]

        # Sort by probability descending
        sorted_indices = class_ids[np.argsort(probs[class_ids])[::-1]]

        return sorted_indices, probs[sorted_indices]
