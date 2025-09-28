# Chapter 2: Trigonometry Refresher

## 2.1 Angles and the Unit Circle

- A **unit circle** is a circle of radius 1 centered at the origin (0,0).  
- Any point on the circle can be described by an angle θ from the x-axis.  

Coordinates of a point at angle θ:
- x = cos(θ)  
- y = sin(θ)  

**Key values:**
- cos(0°) = 1, sin(0°) = 0  
- cos(90°) = 0, sin(90°) = 1  
- cos(180°) = -1, sin(180°) = 0  
- cos(270°) = 0, sin(270°) = -1  

So cosine = horizontal component, sine = vertical component.

---

## 2.2 Sine, Cosine, Tangent

- **Sine (sin θ)** = opposite / hypotenuse  
- **Cosine (cos θ)** = adjacent / hypotenuse  
- **Tangent (tan θ)** = opposite / adjacent = sin θ / cos θ  

These come from right triangles.  

---

## 2.3 Relationship to Vectors

- Cosine gives the **projection of one vector onto another**.  
- That’s why the dot product formula has `cos(θ)`.  
- In embeddings:  
  - If vectors point in the same direction → θ small → cos ≈ 1  
  - If vectors are perpendicular → θ = 90° → cos = 0  
  - If vectors point opposite → θ = 180° → cos = -1  

---

## 2.4 Law of Cosines (Optional)

For a triangle with sides a, b, c, and angle θ opposite side c:

c² = a² + b² − 2ab cos(θ)

This is essentially the dot product formula written in triangle form.

---

## 2.5 Python Examples

```python
import numpy as np

# Angles in radians (np trig uses radians)
angles = [0, np.pi/2, np.pi, 3*np.pi/2]

for a in angles:
    print(f"Angle {a:.2f} rad → cos={np.cos(a):.2f}, sin={np.sin(a):.2f}")
