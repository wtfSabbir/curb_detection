import cv2
import numpy as np
import json
from shapely.geometry import LineString

def extract_geojson_from_heatmap(image_path, output_geojson_path):
    print(f"Loading image: {image_path}")
    # 1. Load the image
    img = cv2.imread(image_path)
    
    # 2. Thresholding: Convert to grayscale
    # Because LISA colored the high-confidence areas bright yellow/green, 
    # they will be the brightest pixels in grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply a binary threshold (Turn everything bright into pure white, everything dark into pure black)
    # We use 150 as a midpoint threshold, but this can be adjusted.
    # 2. Thresholding (Slightly stricter to remove noise)
    _, binary_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # 3. Edge Extraction
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 4. Polyline Simplification
    epsilon = 2.0  
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, closed=True)
    
    # NEW: "Snip the Rubber Band" Logic
    h, w = img.shape[:2]
    margin = 5 # 5 pixels from the edge
    
    segments = []
    current_segment = []
    
    # Walk along the perimeter. If we hit the edge of the photo, cut the line.
    for point in simplified_contour:
        x, y = int(point[0][0]), int(point[0][1])
        if x <= margin or x >= w - margin or y <= margin or y >= h - margin:
            if len(current_segment) > 1:
                segments.append(current_segment)
            current_segment = [] # Reset for the next segment
        else:
            current_segment.append([x, y])
            
    if len(current_segment) > 1:
        segments.append(current_segment)
        
    # The longest remaining segment is our curb line!
    if not segments:
        print("No valid line found inside the image borders.")
        return
        
    points = max(segments, key=len)

    # Draw the cleaned-up line
    cv2.polylines(img, [np.array(points)], isClosed=False, color=(0, 0, 255), thickness=3)
    cv2.imwrite("vectorized_result_clean.jpg", img)
    print("Saved visual result to 'vectorized_result.jpg'")

    # 5. Export to GeoJSON
    # In a production environment, these X,Y pixels would be mapped to real-world coordinates.
    # For this PoC, we are saving the pixel coordinates.
    cad_coordinates = [[float(point[0]), float(point[1]), 0.0] for point in points]

    geojson_data = {
        "type": "FeatureCollection",
        "name": "ai_extracted_curbs",
        "crs": { 
            "type": "name", 
            "properties": { "name": "urn:ogc:def:crs:EPSG::3945" } 
        },
        "features": [
            {
                "type": "Feature",
                "properties": { 
                    "Layer": "VOTROTTO_FE", 
                    "PaperSpace": None, 
                    "SubClasses": "AcDbEntity:AcDbPolyline", 
                    "Linetype": None, 
                    "EntityHandle": "AI_GEN", 
                    "Text": None 
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": cad_coordinates
                }
            }
        ]
    }

    with open(output_geojson_path, 'w') as f:
        json.dump(geojson_data, f, indent=4)
        
    print(f"✅ Successfully exported CAD-ready GeoJSON to {output_geojson_path}")

# Run the function
extract_geojson_from_heatmap("heatmap.jpg", "curb_line.geojson")