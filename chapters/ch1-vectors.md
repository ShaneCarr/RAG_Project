# Chapter 1: Vectors and Linear Algebra

## 1.1 What is a Vector?
- A vector is just an ordered list of numbers.  
- Think of it as an **arrow** pointing in space.  
- Examples:
  - In 2D: [3, 4] → an arrow 3 units right, 4 units up.  
  - In 3D: [1, 2, -1] → an arrow in 3D space.  

Vectors represent:
- Position (where something is)
- Direction (where it points)
- Features (ML: embedding dimensions)

---

## 1.2 Magnitude (Length of a Vector)
The length of vector **v** =

$$
\lVert v \rVert = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
$$


### Python Example
```python
import numpy as np

v = np.array([3, 4])
magnitude = np.linalg.norm(v)
print("Vector:", v)
print("Magnitude:", magnitude)  # Expect 5
```

## 1.3 Vector Addition and Scaling

**Addition**: add components element-wise  
\[
[1, 2] + [3, 4] = [4, 6]
\]

**Scaling**: multiply each component by a scalar  
\[
2 \times [3, 4] = [6, 8]
\]

### Python Example
```python
import numpy as np

u = np.array([1, 2])
v = np.array([3, 4])

print("u + v =", u + v)
print("2 * v =", 2 * v)

```

## 1.4 Dot Product (Similarity)

The **dot product** is a way of measuring how much two vectors align with each other.

### Algebra Definition
\[
u \cdot v = \sum_i u_i v_i
\]

This means: multiply corresponding components and add them up.

### Geometric Definition
\[
u \cdot v = |u| \, |v| \, \cos(\theta)
\]

- If θ = 0° (same direction) → large positive value.  
- If θ = 90° (perpendicular) → 0.  
- If θ = 180° (opposite directions) → negative value.  

### Python Example
```python
import numpy as np

u = np.array([2, 3])
v = np.array([1, 5])

dot = np.dot(u, v)
print("Dot product:", dot)

```
## 1.5 Cosine Similarity

Cosine similarity is the **normalized dot product**. It returns a number in **[-1, 1]** and measures **directional alignment** (ignores length).

**Formula (plain text):**
cosine_sim(u, v) = (u · v) / (||u|| * ||v||)

where
- u · v = u1*v1 + u2*v2 + … + un*vn
- ||u|| = sqrt(u1^2 + u2^2 + … + un^2)
- ||v|| = sqrt(v1^2 + v2^2 + … + vn^2)

**Interpretation**
- Same direction → cosine_sim = 1  
- Perpendicular → cosine_sim = 0  
- Opposite direction → cosine_sim = -1

**ASCII fallback (if you prefer no symbols):**
cosine_sim(u, v) = (dot(u, v)) / (norm(u) * norm(v))

### Python Example
```python
import numpy as np

u = np.array([2, 3])
v = np.array([1, 5])

cosine_sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
print("Cosine similarity:", cosine_sim)
```

## 1.6 Visualization

In 2D, vectors can be drawn as arrows starting at the origin (0,0).

- The **dot product** measures how much one vector "shadows" onto another.
- **Cosine similarity** measures the angle between them, ignoring length.

**Intuition**
- If two arrows point in the same direction → cosine similarity = 1  
- If they are at right angles → cosine similarity = 0  
- If they point in opposite directions → cosine similarity = -1  

### How to Visualize in Python (with matplotlib)
```python
import matplotlib.pyplot as plt
import numpy as np

u = np.array([2, 1])
v = np.array([1, 3])

# Plot vectors
plt.quiver(0, 0, u[0], u[1], angles='xy', scale_units='xy', scale=1, color='blue', label="u")
plt.quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='red', label="v")

plt.xlim(0, 4)
plt.ylim(0, 4)
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.gca().set_aspect('equal')
plt.legend()
plt.title("Visualization of Vectors u and v")
plt.show()
```


## 1.7 Exercises

Try these to reinforce your understanding of vectors:

1. Compute the **magnitude** of the vector `[5, 12]`.  
   *(Hint: use the Pythagorean theorem — expect an integer result.)*

2. Add the vectors `[1, 0, -1]` and `[2, 3, 4]`.  
   *(Check that addition is element-wise.)*

3. Find the **dot product** of `[1, 2]` and `[-2, 1]`.  
   - What does the result tell you about the angle between them?  

4. Compute the **cosine similarity** of `[1, 0]` and `[0, 1]`.  
   - What value do you expect, and why?  

5. (Challenge) Write a Python function that takes two vectors and returns:
   - Their magnitudes
   - Their dot product
   - Their cosine similarity
