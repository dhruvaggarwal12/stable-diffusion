from PIL import Image
import torch
from diffusers import StableDiffusionPipeline
from customtkinter import CTkImage
import tkinter as tk
import customtkinter as ctk
from authtoken import auth_token
from PIL import ImageTk

# Initialize the model
modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

# Entry field
prompt = ctk.CTkEntry(master=app, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
prompt.place(x=10, y=10)

# Place to display image
lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

def generate():
    try:
        # Generate image
        with torch.autocast(device):
            output = pipe(prompt.get(), guidance_scale=8.5)
        
        # Check for NSFW content
        if 'nsfw_content_detected' in output and output['nsfw_content_detected']:
            print("NSFW content detected. Try a different prompt.")
            return
        
        images = output.get('images', [])
        if not images:
            print("No images generated.")
            return
        
        image = images[0]
        
        # Convert PIL Image to CTkImage
        ctk_image = CTkImage(image)
        
        # Save and display the image
        image.save('generatedimage.png')
        img = ImageTk.PhotoImage(image)
        lmain.configure(image=img)
        lmain.image = img  # Keep a reference to avoid garbage collection
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Button to trigger image generation
trigger = ctk.CTkButton(master=app, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=generate)
trigger.configure(text="Generate")
trigger.place(x=206, y=60)

app.mainloop()
