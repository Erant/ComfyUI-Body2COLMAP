# ComfyUI-Body2COLMAP

Generate multi-view training data for **3D Gaussian Splatting** from SAM-3D-Body mesh reconstructions.

This node pack seamlessly integrates with [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) to generate:
- 🎬 Multi-view rendered images from orbit camera paths
- 📷 COLMAP format camera parameters (intrinsics + extrinsics)
- ☁️ Initial point clouds for 3D Gaussian Splatting training

## 🌟 Features

- **3 Camera Path Patterns**: Circular, Sinusoidal, and Helical (recommended for 3DGS)
- **Multiple Render Modes**: Mesh, depth maps, skeleton overlays, and composites
- **COLMAP Export**: Direct export to COLMAP sparse reconstruction format
- **Seamless Integration**: Works directly with SAM3DBody outputs, no file I/O needed
- **Flexible Workflows**: Mix and match nodes with ComfyUI's built-in SaveImage, PreviewImage, etc.

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)

1. Open ComfyUI Manager
2. Search for "Body2COLMAP"
3. Click Install

### Method 2: Manual Installation

```bash
# 1. Install body2colmap package
cd /path/to/body2colmap
pip install -e .

# 2. Install node pack
cd ComfyUI/custom_nodes/
git clone https://github.com/your-repo/ComfyUI-Body2COLMAP.git
cd ComfyUI-Body2COLMAP
pip install -r requirements.txt
```

### Prerequisites

- **[body2colmap](https://github.com/Erant/body2colmap)**: Core rendering library (install with `pip install -e /path/to/body2colmap`)
- **[ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody)**: Generates 3D mesh from single image

## 🚀 Quick Start

### Basic Workflow: Generate 3DGS Training Data

```
Load Image → SAM3DBodyProcess → Helical Path → Render → SaveImage
                                                      ↓
                                                ExportCOLMAP
```

1. **Load Image**: Use ComfyUI's built-in LoadImage node
2. **SAM3DBodyProcess**: Generate 3D mesh from single image (from ComfyUI-SAM3DBody)
3. **Helical Path**: Generate helical camera orbit (120 frames, 3 loops, 30° amplitude)
4. **Render**: Render multi-view images (512x512, mesh mode)
5. **SaveImage**: Save rendered frames (prefix: "frame_")
6. **ExportCOLMAP**: Export camera parameters and point cloud to `output/colmap/`

**Result**: Ready-to-use training data for 3D Gaussian Splatting!

## 📚 Nodes Reference

### 🌀 Path Generators

#### Helical Path (3DGS) - Recommended

Generates multi-loop helical path with smooth elevation changes. Best for 3D Gaussian Splatting training.

**Parameters:**
- `n_frames` (120): Total number of camera positions
- `n_loops` (3): Number of full 360° rotations
- `amplitude_deg` (30°): Elevation range (-30° to +30°)
- `lead_in_deg` (45°): Rotation at bottom before ascending
- `lead_out_deg` (45°): Rotation at top after ascending

#### Circular Path

Simple circular orbit at fixed elevation. Good for turntable-style views.

**Parameters:**
- `n_frames` (36): Number of camera positions
- `elevation_deg` (0°): Camera elevation angle
- `start_azimuth_deg` (0°): Starting rotation angle

#### Sinusoidal Path

Orbit with oscillating elevation. Better coverage for complex shapes.

**Parameters:**
- `n_frames` (60): Number of camera positions
- `amplitude_deg` (30°): Maximum elevation deviation
- `n_cycles` (2): Number of up/down oscillations

**Common optional parameters for all paths:**
- `radius` (0=auto): Orbit radius in meters
- `fill_ratio` (0.8): How much of viewport contains mesh
- `width/height` (512): Image dimensions for auto-radius
- `focal_length` (0=auto): Camera focal length in pixels

### 🎬 Render Multi-View

Renders images from all camera positions in the path.

**Parameters:**
- `width/height` (512): Output image dimensions
- `render_mode`: Choose from:
  - `mesh`: Solid color mesh rendering
  - `depth`: Depth map visualization
  - `skeleton`: Skeleton keypoints only
  - `mesh+skeleton`: Mesh with skeleton overlay
  - `depth+skeleton`: Depth map with skeleton overlay

**Mesh options:**
- `mesh_color_r/g/b`: Mesh color (default: light blue 0.65, 0.74, 0.86)
- `bg_color_r/g/b`: Background color (default: white 1.0, 1.0, 1.0)

**Skeleton options:**
- `skeleton_format`: `openpose_body25_hands` or `mhr70`
- `joint_radius` (0.015): Joint sphere size in meters
- `bone_radius` (0.008): Bone cylinder size in meters

**Depth options:**
- `depth_colormap`: `grayscale`, `viridis`, `plasma`, `inferno`, `magma`

**Outputs:**
- `images`: Batch tensor [N, H, W, 3] for SaveImage/PreviewImage
- `render_data`: Metadata for ExportCOLMAP node

### 📦 Export COLMAP

Exports COLMAP sparse reconstruction format.

**Parameters:**
- `output_directory` ("output/colmap"): Output path
- `filename_pattern` ("frame_{:04d}.png"): Must match SaveImage prefix
- `pointcloud_samples` (50000): Number of surface points to sample
- `save_images` (True): Also save images to `images/` subdirectory

**Output structure:**
```
output/colmap/
├── images/              (if save_images=True)
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
└── sparse/0/
    ├── cameras.txt      (camera intrinsics)
    ├── images.txt       (camera extrinsics per frame)
    └── points3D.txt     (initial point cloud)
```

## 🎯 Example Workflows

### Portrait Mode for Mobile 3DGS

Generate vertical images optimized for mobile viewing:

```
SAM3DBodyProcess → HelicalPath → Render (720x1280) → SaveImage
                                                   ↓
                                             ExportCOLMAP
```

- Resolution: 720x1280 (portrait)
- Path: Helical, 120 frames, 3 loops, 30° amplitude
- Auto-framing handles portrait aspect ratio correctly

### Compare Multiple Path Types

Preview different camera paths side-by-side:

```
SAM3DBodyProcess ─┬→ CircularPath → Render → SaveAnimatedWEBP
                  ├→ SinusoidalPath → Render → SaveAnimatedWEBP
                  └→ HelicalPath → Render → SaveAnimatedWEBP
```

Creates three animated previews showing different orbit patterns.

### Debug Skeleton Alignment

Verify skeleton fitting with overlay:

```
SAM3DBodyProcess → HelicalPath → Render (mesh+skeleton) → SaveImage
                                                        ↓
                                                  PreviewImage
```

- Mode: `mesh+skeleton` or `depth+skeleton`
- Adjust `joint_radius` and `bone_radius` for visibility

## 🔧 Advanced Usage

### Custom Focal Length

Override auto-computed focal length for specific FOV:

```python
# 35mm equivalent on full-frame: ~63° FOV
focal_length = width / (2 * tan(radians(63/2)))
```

Set `focal_length` in both path generator (optional) and render node.

### Match Existing Image Filenames

If using separate SaveImage node:

1. SaveImage prefix: `training_data_`
2. ExportCOLMAP filename_pattern: `training_data_{:05d}.png`

Pattern must match for COLMAP to find images.

### High-Resolution Output

For production 3DGS models:

- Resolution: 1920x1080 or higher
- Frames: 200-300 for complex scenes
- Pattern: Helical with 4-5 loops
- Point cloud samples: 100,000+

## 📖 Tips & Best Practices

### For 3D Gaussian Splatting:

1. **Use Helical Path**: Best multi-view coverage with smooth motion
2. **Frame Count**: 120-200 frames for good reconstruction
3. **Elevation Range**: 30-45° covers most important angles
4. **Resolution**: Match your target viewing resolution
5. **Fill Ratio**: 0.7-0.8 ensures mesh stays in frame during orbit

### For General Use:

- **Preview First**: Use PreviewImage or SaveAnimatedWEBP (24 frames) before full render
- **Portrait vs Landscape**: Auto-framing adapts to aspect ratio
- **Skeleton Debug**: Use `mesh+skeleton` mode to verify pose estimation
- **Depth Maps**: Useful for training depth-aware 3DGS variants

## 🤝 Integration with Other Nodes

### ComfyUI Built-ins

- **SaveImage**: Saves each frame as separate PNG
- **PreviewImage**: Preview entire batch in UI
- **SaveAnimatedWEBP**: Create animated preview
- **ImageFromBatch**: Extract single frame for preview

### Compatible Node Packs

- **ComfyUI-SAM3DBody** (required): Source of mesh data
- **ComfyUI-3D-Pack**: Further 3D processing
- **ComfyUI-VideoHelperSuite**: Video export of rendered sequences

## 🐛 Troubleshooting

### "SAM3D_OUTPUT type not found"

Install [ComfyUI-SAM3DBody](https://github.com/PozzettiAndrea/ComfyUI-SAM3DBody) first.

### "Unable to create GL context" (headless)

Install OSMesa for software rendering:
```bash
# Ubuntu/Debian
sudo apt-get install libosmesa6-dev

# Set environment variable
export PYOPENGL_PLATFORM=osmesa
```

### "Mesh appears tiny in frame"

- Increase `fill_ratio` (try 0.9)
- Check `radius=0` for auto-compute
- Verify path node has correct `width/height`

### "COLMAP can't find images"

Ensure `filename_pattern` in ExportCOLMAP matches SaveImage prefix:
- SaveImage prefix: `frame_`
- ExportCOLMAP pattern: `frame_{:04d}.png`

## 📝 License

MIT License - see LICENSE file for details.

## 🙏 Credits

- **body2colmap**: Core rendering and COLMAP export library
- **ComfyUI-SAM3DBody**: SAM-3D-Body mesh reconstruction
- **pyrender**: OpenGL rendering backend
- **COLMAP**: Structure-from-Motion pipeline

## 📮 Support & Contributing

- **Issues**: [GitHub Issues](https://github.com/your-repo/ComfyUI-Body2COLMAP/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/ComfyUI-Body2COLMAP/discussions)
- **Contributing**: Pull requests welcome!

---

**Happy 3D Gaussian Splatting! 🌟**
