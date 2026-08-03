from __future__ import annotations
import os
import uuid
import numpy as np
from PIL import Image, ImageDraw

def generate_synthetic_fundus(condition: str, severity: str = "moderate") -> Tuple[str, str]:
    """
    Generates a realistic mock fundus retinal photograph with simulated pathological features,
    and returns (image_file_path, heatmap_file_path).
    
    Acts as the AI-Powered Synthetic Case Generator for junior clinicians/students.
    """
    # Create retinal base: 512x512 RGB
    img = Image.new("RGB", (512, 512), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    
    # Base retina (orange/red disk)
    draw.ellipse([64, 64, 448, 448], fill=(180, 70, 30))
    
    # Macula (darker spot in center)
    draw.ellipse([230, 230, 282, 282], fill=(130, 45, 20))
    
    # Optic Disc (yellow/white circle on the side)
    draw.ellipse([340, 210, 410, 280], fill=(245, 220, 140))
    
    # Retinal arcade vessels (curved lines radiating from optic disc)
    vessel_color = (130, 15, 10)
    draw.arc([150, 80, 370, 240], start=30, end=170, fill=vessel_color, width=3)
    draw.arc([150, 240, 370, 400], start=190, end=330, fill=vessel_color, width=3)
    draw.line([375, 245, 300, 160], fill=vessel_color, width=2)
    draw.line([375, 245, 300, 320], fill=vessel_color, width=2)
    
    # Add pathology indicators based on condition
    if condition == "Cataract":
        # Cloudiness in center (pupil opacity)
        opacity = 180 if severity == "severe" else 100
        draw.ellipse([180, 180, 332, 332], fill=(220, 220, 220), outline=(240, 240, 240))
        # Draw cloudy layer
        cloud = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        c_draw = ImageDraw.Draw(cloud)
        c_draw.ellipse([200, 200, 312, 312], fill=(255, 255, 255, opacity))
        img = Image.alpha_composite(img.convert("RGBA"), cloud).convert("RGB")
        draw = ImageDraw.Draw(img)
        
    elif condition == "Conjunctivitis":
        # Red scleral injection at edges
        for i in range(0, 360, 15):
            rad = np.radians(i)
            x1 = 256 + 185 * np.cos(rad)
            y1 = 256 + 185 * np.sin(rad)
            x2 = 256 + 215 * np.cos(rad)
            y2 = 256 + 215 * np.sin(rad)
            draw.line([x1, y1, x2, y2], fill=(200, 30, 20), width=2)
            
    elif condition == "Uveitis":
        # Inflammatory cell deposits (white dots/mutton-fat keratic precipitates)
        for _ in range(25):
            cx = np.random.randint(150, 350)
            cy = np.random.randint(150, 350)
            r = np.random.randint(3, 7)
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=(235, 235, 210))
            
    elif condition == "Jaundice":
        # Scleral yellowing at outer ring
        yellow_mask = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
        y_draw = ImageDraw.Draw(yellow_mask)
        y_draw.ellipse([50, 50, 462, 462], outline=(230, 200, 20, 150), width=18)
        img = Image.alpha_composite(img.convert("RGBA"), yellow_mask).convert("RGB")
        draw = ImageDraw.Draw(img)
        
    elif condition == "Eyelid":
        # Partial occlusion by eyelid droop (ptosis/chalazion mock)
        draw.polygon([(0, 0), (512, 0), (512, 140), (0, 100)], fill=(80, 50, 40))
        
    elif condition == "Pterygium":
        # Fleshy growth invading cornea from inner corner
        draw.polygon([(64, 256), (180, 220), (220, 256), (180, 292)], fill=(210, 170, 160), outline=(220, 120, 100))

    # Save to dynamic storage paths
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
    os.makedirs(static_dir, exist_ok=True)
    
    case_id = f"synthetic-{uuid.uuid4().hex[:8]}"
    img_name = f"{case_id}.jpg"
    heatmap_name = f"{case_id}_cam.jpg"
    
    img_path = os.path.join(static_dir, img_name)
    img.save(img_path, "JPEG")
    
    # Generate matching Grad-CAM heatmap
    heatmap = Image.new("RGB", (512, 512), color=(0, 0, 0))
    h_draw = ImageDraw.Draw(heatmap)
    # Focus Grad-CAM where lesions were drawn
    if condition == "Cataract":
        h_draw.ellipse([190, 190, 320, 320], fill=(255, 0, 0))
    elif condition == "Uveitis":
        h_draw.ellipse([200, 220, 300, 320], fill=(255, 80, 0))
    elif condition == "Pterygium":
        h_draw.ellipse([100, 220, 210, 280], fill=(255, 0, 0))
    else:
        h_draw.ellipse([220, 220, 290, 290], fill=(255, 120, 0))
        
    heatmap_path = os.path.join(static_dir, heatmap_name)
    heatmap.save(heatmap_path, "JPEG")
    
    # Return virtual asset URLs/relative paths
    return f"/static/{img_name}", f"/static/{heatmap_name}"
