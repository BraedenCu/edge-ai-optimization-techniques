# Demo Ideas: Optimizing On-Device Inference for Diffusion & LLM Models

Below is a list of demo ideas that integrate key components from the provided resources. Each demo highlights different optimization techniques from the paper, along with complementary tools from the AI Edge Torch Generative API, MediaPipe LLM Inference, and XNNPACK.

---

## 1. End-to-End Diffusion Model Inference Demo

**What to do:**
- Deploy a diffusion model (e.g., Stable Diffusion 1.4) on a mobile device (or simulated mobile environment) using the AI Edge Torch pipeline and TF Lite runtime.
- Toggle optimization options such as specialized fused kernels for Group Norm/GELU, partially fused softmax, FlashAttention, and Winograd convolution.
- Measure and display inference latency and memory usage differences as you enable/disable each optimization.

**Key Components:**
- GPU-Aware optimizations from the paper (fused kernels, softmax optimizations, Winograd convolution).
- On-device execution via TF Lite with the XNNPACK delegate.

---

## 2. Attention Module Optimization Visualization

**What to do:**
- Implement two versions of the attention module: a baseline using standard softmax and an optimized version using partially fused softmax (and optionally FlashAttention for certain cases).
- Create a dashboard that visualizes per-layer latency, memory consumption, and throughput for each version.
- Allow users to interactively switch between modes to see real-time performance improvements.

**Key Components:**
- Attention module optimization techniques (fused softmax reduction, FlashAttention).
- Benchmarking insights similar to those presented in Table 1 of the paper.

---

## 3. Winograd Convolution Tuner

**What to do:**
- Develop a demo that replaces standard 3×3 convolutions in a key module (e.g., the image decoder) with a configurable Winograd convolution layer.
- Allow users to adjust the tile size (2×2, 4×4, 6×6, 8×8) and observe the trade-offs between computational savings and memory overhead.
- Present performance metrics (FLOPs reduction, latency, and memory usage) to highlight the optimal configuration.

**Key Components:**
- Winograd convolution optimizations as described in the paper.
- Integration with XNNPACK, which provides low-level acceleration.

---

## 4. Custom LLM Inference Demo with Multi-Signature Conversion

**What to do:**
- Re-author a compact transformer (e.g., TinyLlama) using the AI Edge Torch Generative API building blocks.
- Convert the model to a multi-signature TFLite model that separates prefill and decode functions, incorporating quantization and composite op lowering for performance.
- Deploy the model using the MediaPipe LLM Inference API and compare inference speed and resource usage with/without these optimizations.

**Key Components:**
- The AI Edge Torch authoring and conversion flow (including weight remapping, quantization APIs, and multi-signature export).
- On-device acceleration via TF Lite/XNNPACK delegate, emphasizing weight caching and composite op lowering.

---

## 5. Benchmarking Dashboard for Incremental Optimizations

**What to do:**
- Build a dashboard that sequentially applies optimization techniques (starting from a baseline implementation).
- For each stage (e.g., baseline, + fused softmax, + fused Group Norm/GELU, + FlashAttention, + Winograd), display metrics such as latency, memory usage for tensors and weights, and overall end-to-end inference time.
- Use graphs or tables to visualize how each optimization contributes to performance improvements.

**Key Components:**
- The incremental performance improvements detailed in the paper’s Table 1.
- Integration of metrics from both the TF Lite/XNNPACK side and the AI Edge Torch conversion pipeline.

---

# First Steps: Local Optimization & Mobile Deployment Demo

This section details every step required to perform local optimizations in a Jupyter Notebook and then deploy a Stable Diffusion model—fine-tuned to generate images of dogs—on a mobile application. We also include instructions for precise latency tracking of both the unoptimized and optimized models.

--- 


# First Steps: Local Optimization & Mobile Deployment Demo

## 1. Environment Setup

1. **Prerequisites:**
   - A machine with a CUDA-capable GPU (for training/fine-tuning and inference profiling).
   - Python 3.8+ installed.
   - [Conda](https://docs.conda.io/en/latest/) or [virtualenv](https://virtualenv.pypa.io/) for creating isolated environments.
   - Jupyter Notebook installed.

2. **Create a Virtual Environment and Install Dependencies:**

   ```bash
   # Using conda:
   conda create -n stable_diffusion_env python=3.9 -y
   conda activate stable_diffusion_env

   # Or using virtualenv:
   python -m venv stable_diffusion_env
   source stable_diffusion_env/bin/activate  # On Windows: stable_diffusion_env\Scripts\activate

   # Install required packages
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
   pip install diffusers accelerate transformers
   pip install jupyterlab matplotlib numpy
   pip install tensorflow tensorflow-lite  # For later conversion to TFLite
   pip install ai-edge-torch  # Hypothetical package for AI Edge Torch Generative API


3. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

---

## 2. Data Preparation

1. **Obtain a Dog Images Dataset:**
   - Download a dog images dataset (e.g., [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)) or use a Kaggle dataset.
   - Organize the dataset into folders (e.g., `data/dogs/`) for use during fine-tuning.

2. **Preprocess the Data:**
   - Resize images to the resolution expected by your diffusion model (e.g., 512×512 pixels).
   - (Optional) Create a custom PyTorch `Dataset` and `DataLoader` for your training loop.

---

## 3. Fine-Tuning Stable Diffusion on Dog Images

---

1. **Begin Optional Custom Model Finetuning**

In this example, we use the Oxford-IIIT Pet Dataset—a well-known dataset containing images of various pets (with a strong focus on dogs and cats). You can download it from [here](http://www.robots.ox.ac.uk/~vgg/data/pets/).

## Fine-Tuning Stable Diffusion on Dog Images

In this section, we will fine-tune a pre-trained Stable Diffusion model on a dataset of dog images. We will use the [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) as our source. (Note that while the dataset contains both dogs and cats, you can filter or label the images to only use dog images during training.)

### Steps for Fine-Tuning

1. **Download and Prepare the Dataset:**
   - Download the Oxford-IIIT Pet Dataset from [this link](http://www.robots.ox.ac.uk/~vgg/data/pets/).
   - Extract the images and annotations.
   - (Optional) Filter the dataset to only include dog images. You can do this by using the provided annotations or by manually selecting folders corresponding to dog breeds.
   - Resize the images to the target resolution (e.g., 512×512 pixels). You can use a simple Python script with PIL or OpenCV:

   ```python
     from PIL import Image
     import os

     input_folder = "path/to/extracted/images"
     output_folder = "data/dogs"
     os.makedirs(output_folder, exist_ok=True)

     for file_name in os.listdir(input_folder):
         if file_name.endswith(".jpg"):
             image_path = os.path.join(input_folder, file_name)
             img = Image.open(image_path).convert("RGB")
             img = img.resize((512, 512))
             # Optionally, filter based on breed or use a pre-built classifier to select dog images.
             img.save(os.path.join(output_folder, file_name))
     ```

2. **Set Up Your Fine-Tuning Environment in a Jupyter Notebook:**

   Ensure you have installed all necessary packages (see Environment Setup in the previous section).

3. **Load the Pre-trained Stable Diffusion Model:**

   ```python
   from diffusers import StableDiffusionPipeline
   import torch

   # Load pre-trained model (ensure you have sufficient GPU memory)
   model_id = "CompVis/stable-diffusion-v1-4"
   pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
   pipe.to("cuda")
   ```

4. **Prepare a Custom Dataset and DataLoader:**

   Create a PyTorch dataset to load images from your dog images folder:

   ```python
   import os
   from PIL import Image
   from torch.utils.data import Dataset, DataLoader
   import torchvision.transforms as transforms

   class DogDataset(Dataset):
       def __init__(self, image_folder, transform=None):
           self.image_folder = image_folder
           self.image_files = [os.path.join(image_folder, file)
                               for file in os.listdir(image_folder) if file.endswith(".jpg")]
           self.transform = transform

       def __len__(self):
           return len(self.image_files)

       def __getitem__(self, idx):
           img_path = self.image_files[idx]
           img = Image.open(img_path).convert("RGB")
           if self.transform:
               img = self.transform(img)
           return img

   transform = transforms.Compose([
       transforms.Resize((512, 512)),
       transforms.ToTensor(),
       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
   ])

   dataset = DogDataset("data/dogs", transform=transform)
   dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
   ```

5. **Define a Fine-Tuning Training Loop:**

   You can now create a training loop. Note that fine-tuning diffusion models requires adjusting both the denoising network and possibly the text conditioning (if you’re using text prompts). For simplicity, the following example shows a basic training loop outline:

   ```python
   import torch.optim as optim

   # Define optimizer (you might want to use a lower learning rate for fine-tuning)
   optimizer = optim.Adam(pipe.unet.parameters(), lr=1e-5)

   num_epochs = 3  # Adjust epochs as needed
   device = "cuda"

   pipe.unet.train()  # Set the UNet to training mode

   for epoch in range(num_epochs):
       for batch in dataloader:
           # Move batch to device
           batch = batch.to(device)
           
           # Sample random noise and timesteps for diffusion training
           noise = torch.randn(batch.shape, device=device)
           timesteps = torch.randint(0, 1000, (batch.shape[0],), device=device)

           # Predict noise using the UNet of the diffusion pipeline
           loss = pipe.unet(batch, timesteps, noise).loss

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           print(f"Epoch: {epoch}, Loss: {loss.item()}")

   # After training, save a fine-tuned checkpoint:
   fine_tuned_checkpoint = "fine_tuned_dog_model"
   pipe.save_pretrained(fine_tuned_checkpoint)
   ```

6. **Reload the Fine-Tuned Model:**

   ```python
   # Reload the fine-tuned model for inference
   pipe = StableDiffusionPipeline.from_pretrained(fine_tuned_checkpoint, torch_dtype=torch.float16)
   pipe.to("cuda")
   ```

7. **Test Generation on Dog Prompts:**

   ```python
   prompt = "A cute dog playing in a sunny park"
   image = pipe(prompt).images[0]
   image.save("fine_tuned_dog.png")
   ```

### Dataset Recommendation

For this fine-tuning demo, we recommend using the **Oxford-IIIT Pet Dataset**. It is freely available and contains a diverse set of pet images, including a large number of dog images. Download it from:  
[http://www.robots.ox.ac.uk/~vgg/data/pets/](http://www.robots.ox.ac.uk/~vgg/data/pets/)

1. **END CUSTOM FINETUNING** 

---

1. **Load a Pre-trained Stable Diffusion Model:**
   - Use the [diffusers](https://huggingface.co/docs/diffusers/index) library to load a pre-trained checkpoint.

2. **Fine-Tune the Model:**

   In a new notebook cell, run:

   ```python
   from diffusers import StableDiffusionPipeline
   import torch

   # Load pre-trained model (ensure you have sufficient GPU memory)
   model_id = "CompVis/stable-diffusion-v1-4"
   pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
   pipe.to("cuda")

   # (Optional) Modify the pipeline for fine-tuning on dog images
   # Define your custom training loop here to fine-tune on images from data/dogs/
   # For demonstration purposes, assume you have a fine-tuning script that saves a checkpoint:
   fine_tuned_checkpoint = "path/to/fine_tuned_dog_model"

   # Reload the fine-tuned model
   pipe = StableDiffusionPipeline.from_pretrained(fine_tuned_checkpoint, torch_dtype=torch.float16)
   pipe.to("cuda")
   ```

3. **Test Generation (Baseline):**

   ```python
   prompt = "A cute dog playing in a sunny park"
   image = pipe(prompt).images[0]
   image.save("baseline_dog.png")
   ```

---

## 4. Applying Optimizations Locally

To simulate the GPU-aware optimizations from the paper (e.g., specialized fused kernels, partially fused softmax, FlashAttention, and Winograd convolution), modify the inference pipeline as follows:

1. **Enable Optimized Kernels:**
   - If using a custom diffusion implementation or wrapper, set flags or pass parameters to enable optimizations. For example:

   ```python
   # Hypothetical flag to enable optimizations in our pipeline
   pipe.enable_optimizations = True
   ```

2. **Run Inference with Optimizations:**

   ```python
   optimized_image = pipe(prompt).images[0]
   optimized_image.save("optimized_dog.png")
   ```

---

## 5. Latency Tracking in Jupyter Notebook

1. **Measure Inference Latency (Unoptimized):**

   ```python
   import time

   # Warm-up run
   _ = pipe(prompt)

   # Measure unoptimized latency
   torch.cuda.synchronize()
   start = time.perf_counter()
   _ = pipe(prompt)
   torch.cuda.synchronize()
   unoptimized_latency = time.perf_counter() - start
   print(f"Unoptimized Inference Latency: {unoptimized_latency * 1000:.2f} ms")
   ```

2. **Measure Inference Latency (Optimized):**

   ```python
   # Ensure optimizations are enabled
   pipe.enable_optimizations = True

   # Warm-up run
   _ = pipe(prompt)

   # Measure optimized latency
   torch.cuda.synchronize()
   start = time.perf_counter()
   _ = pipe(prompt)
   torch.cuda.synchronize()
   optimized_latency = time.perf_counter() - start
   print(f"Optimized Inference Latency: {optimized_latency * 1000:.2f} ms")
   ```

3. **Log and Compare Results:**
   - Document the latency values (e.g., unoptimized ~1100 ms, optimized ~525 ms per inference iteration) as observed during tests.

---

## 6. Converting the Model for Mobile Deployment

1. **Export the Model using AI Edge Torch Generative API:**

   ```python
   import ai_edge_torch
   # Define input tensors for conversion (e.g., prefill tokens)
   prefill_tokens = torch.zeros((1, 512), dtype=torch.long, device="cpu")
   prefill_positions = torch.arange(512, device="cpu")

   # Convert the fine-tuned model to a multi-signature TFLite model
   edge_model = (
       ai_edge_torch.signature("prefill", pipe, (prefill_tokens, prefill_positions))
       .signature("decode", pipe, (torch.zeros((1, 1), dtype=torch.long, device="cpu"), torch.tensor([0], device="cpu")))
       .convert(quant_config="full_linear_int8_dynamic_recipe")  # Adjust quantization recipe as needed
   )
   tflite_file = "/tmp/stable_diffusion_dogs.tflite"
   edge_model.export(tflite_file)
   ```

2. **Verify the TFLite Model:**
   - Use TensorFlow Lite’s Python interpreter to run a simple inference test on your exported model.

---

## 7. Running the Model on a Mobile Application

1. **Integrate the TFLite Model into a Mobile App:**
   - For Android:
     - Place the TFLite file (`stable_diffusion_dogs.tflite`) in the `assets` folder.
     - Use the [TensorFlow Lite Android Interpreter API](https://www.tensorflow.org/lite/guide/inference) to load and run the model.
     - Example snippet:

       ```java
       // Java: Load TFLite model from assets
       Interpreter tflite = new Interpreter(loadModelFile("stable_diffusion_dogs.tflite"));
       // Set up input/output buffers and run inference:
       tflite.run(inputBuffer, outputBuffer);
       ```

   - For iOS:
     - Add the TFLite model to the Xcode project.
     - Use the [TensorFlow Lite Swift API](https://www.tensorflow.org/lite/guide/inference_ios) to load and run the model.

2. **Implement Latency Tracking on Mobile:**
   - Instrument the inference call by recording timestamps before and after the `interpreter.run(...)` call.
   - For example, on Android:

     ```java
     long startTime = System.nanoTime();
     tflite.run(inputBuffer, outputBuffer);
     long endTime = System.nanoTime();
     long latencyMs = (endTime - startTime) / 1_000_000;
     Log.d("Latency", "Inference Latency: " + latencyMs + " ms");
     ```

3. **Compare Optimized vs Unoptimized Mobile Inference:**
   - Deploy two versions of the app: one with the baseline (unoptimized) TFLite model and another with the optimized TFLite model.
   - Collect latency measurements using the above instrumentation and display/compare the results (e.g., via on-screen logs or a debugging console).

---

## 8. Summary

- **Local Steps:**  
  - Set up the environment and prepare the dataset.
  - Fine-tune a pre-trained Stable Diffusion model on a dataset of dog images.
  - Apply optimizations locally (fused kernels, optimized attention, Winograd convolution).
  - Track inference latency in the Jupyter Notebook for both unoptimized and optimized models.

- **Mobile Deployment:**  
  - Convert the optimized PyTorch model to a multi-signature TFLite model using the AI Edge Torch Generative API.
  - Integrate the TFLite model into a mobile application (Android/iOS).
  - Use platform-specific tools to track inference latency on device, demonstrating the performance gains.

---
