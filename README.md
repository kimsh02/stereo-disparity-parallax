# Depth Estimation from Stereo Disparity and Astronomical Parallax

This project explores two key applications of disparity-based depth estimation using MATLAB:
1. Generating disparity maps from stereo image pairs to estimate object distances.
2. Using astronomical parallax to calculate the distance to a nearby star.

---

## A. Disparity Map Computation from Stereo Images

This section focuses on computing disparity maps from random dot stereograms using stereo image pairs (`left1/right1` and `left2/right2`).

### Method:
- Maximum disparity: **D = 15**
- Local context window size: **C = 10**
- Total margin: **D + C = 25**
- Disparity computed by:
  - Normalizing row vectors from stereo images.
  - Finding the dot product to determine best-matching disparity.
  - Recording disparity with the highest dot product.
- MATLAB's `imagesc()` function visualizes the results (lighter pixels = higher disparity).

### Results:
- **Average disparity** (top 10% of highest values):
  - Set 1: **10.3035**
  - Set 2: **10.3275**

- **Estimated distances using the parallax depth formula** \( z = \frac{f(d - D)}{D} \):
  - Focal length: **f = 2 cm**, Interocular distance: **d = 10 cm**
  - Pixel size: **p = 0.025 cm**
  - Converted disparity: **D ≈ 0.25 cm**

  | Set | Distance (d ≫ D) | Distance (d − D) |
  |-----|------------------|------------------|
  | 1   | 77.6432 cm       | 75.6432 cm       |
  | 2   | 77.4631 cm       | 75.4631 cm       |

> Note: Both distance methods are acceptable as discussed in class and Piazza.

---

## B. Star Distance Estimation Using Astronomical Parallax

This section uses parallax and known imaging parameters to estimate the distance to a star.

### Parameters:
- Focal length: **f = 2 m**
- Interocular distance (Earth's orbit diameter): **d = 80,000,000 miles = 1.2875×10¹¹ m**
- Measured disparity: **D = 6.2×10⁻⁶ m**

### Calculations:
- \[
z = \frac{f \cdot d}{D} = \frac{2 \cdot 1.2875 \times 10^{11}}{6.2 \times 10^{-6}} = 4.1531 \times 10^{16} \text{ m}
\]
- Converted to miles:
  \[
z \approx 2.5806 \times 10^{13} \text{ miles}
\]
- Converted to light-years:
  \[
z \approx \frac{2.5806 \times 10^{13}}{5.879 \times 10^{12}} \approx 4.39 \text{ light-years}
\]

### Interpretation:
- The estimated distance corresponds closely to **Alpha Centauri** (either Rigil Kentaurus or Toliman).
- Their known distance: **4.3441 ± 0.0022 light-years**.
- These stars form a binary system, often treated as one in early measurements.

---

## View PDF Result
[Open hw13.pdf](./hw13.pdf)
