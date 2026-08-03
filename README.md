# Jink3D

**Interactive stereo-video marking, camera calibration, orientation, and 3D reconstruction**

Jink3D is a Python application for marking corresponding points in two synchronized camera views and reconstructing their three-dimensional coordinates. It displays a frame slider, two adjustable video views, an interactive console, a 3D view of reconstructed trajectories, and the coordinates of visible markers in the current frame.

Jink3D currently supports:

- two synchronized AVI videos;
- up to nine manually marked points;
- spline interpolation between marked frames;
- stereo camera calibration with ChArUco boards, checkerboards, and circle grids;
- gravity alignment from a plumbline or a freely thrown object;
- saving and loading marker data, camera geometry, and orientation;
- export of reconstructed 3D data.

The software is under active development. 

## Installation

The project uses [`uv`](https://docs.astral.sh/uv/) to manage its Python environment and dependencies.

```bash
git clone https://github.com/jamietheobald/jink3d.git
cd jink3d
uv sync
uv run python jink3d.py
```

In PyCharm, use the interpreter created by `uv`:

```text
<project folder>/.venv/bin/python
```

The principal dependencies are NumPy, SciPy, PyQt5, pyqtgraph, PyOpenGL, and the headless OpenCV contrib package. The contrib build is required for ArUco and ChArUco support. The headless build avoids a conflict between OpenCV's bundled Qt plugins and PyQt5.

## Interface

The main window contains:

1. a frame slider and frame number;
2. the left and right video views, each with brightness controls;
3. an interactive Python console;
4. a 3D view of reconstructed tracks and camera positions;
5. a table of marker coordinates for the current frame;
6. **camera** and **orientation** indicators, which turn green when the corresponding calibration has been loaded or created.

The console namespace includes:

```python
st  # the main Jink3D window
pg  # pyqtgraph
np  # NumPy
```

## Loading stereo videos

Choose **File → Open avis...** or press **Ctrl+O**.

The two videos should:

- contain the same number of frames;
- be synchronized;
- show overlapping regions of space;
- remain in the same camera geometry used for calibration.

If the selected filename begins with uppercase `L` or `R`, Jink3D looks in the same directory for a matching file whose first character is replaced by the other letter. For example:

```text
Ltrial_001.avi
Rtrial_001.avi
```

If no matching file is found, Jink3D asks for the second video.

Camera 0 is treated as the left camera and defines the origin of the reconstructed coordinate system. Camera 1 is positioned relative to it by the stereo calibration.

## Navigation

| Action | Shortcut |
|---|---|
| Previous or next frame | **Left** or **Right** |
| Back or forward 5 frames | **Alt+Left** or **Alt+Right** |
| Back or forward 50 frames | **Ctrl+Left** or **Ctrl+Right** |
| Previous or next marked frame | **Shift+Left** or **Shift+Right** |
| Previous or next midpoint between marked frames | **Ctrl+Shift+Left** or **Ctrl+Shift+Right** |
| Reset all views | **Ctrl+R** |
| Reset the view under the pointer | **`** (backquote) |
| Toggle fullscreen | **Ctrl+F** |
| Print keyboard help | **Ctrl+H** |

“Marked frame” navigation includes user markers and frames in which a calibration target was detected. This makes **Shift+Left** and **Shift+Right** useful for inspecting camera-calibration frames.

Midpoint navigation uses only user-marked tracking frames. It jumps to the midpoint between adjacent marked frames, where interpolation error is often easiest to assess.

## Marking points

Jink3D supports markers 1–9. Each marker has its own color.

### Mouse controls

- Choose the active marker from the **Markers** menu, then click in an image to place it.
- Clicking again replaces the marker position in that frame.
- Alt-clicking removes the selected marker.

### Keyboard controls

With the pointer over an image:

| Action | Shortcut |
|---|---|
| Place marker 1–9 | **1–9** |
| Place marker 1–9 and advance 50 frames | **Ctrl+1–9** |
| Remove marker 1–9 | **Alt+1–9** |
| Undo the most recent marker edit | **Ctrl+Z** |

With the pointer over the 3D display, pressing **1–9** pans the 3D view to that marker.

### Interpolation

For each marker, Jink3D fits independent splines to the marked x and y positions in each camera. The spline order increases with the number of marked frames, up to cubic interpolation.

- Directly marked points appear strongly colored.
- Interpolated points appear translucent.
- Jink3D interpolates only between marked frames; it does not extrapolate beyond them.

A useful workflow is:

1. mark relatively widely spaced frames;
2. use **Ctrl+Shift+Left** and **Ctrl+Shift+Right** to inspect interval midpoints;
3. add another marker where the object has departed too far from the interpolation;
4. repeat until the interpolation is adequate.

Marking too many closely spaced points can sometimes produce unnecessary spline oscillation.

A reconstructed 3D position requires a direct or interpolated marker position in both camera views at the same frame.

## Image display

Pan and zoom each camera view with the mouse. The histogram controls beside each image adjust the displayed intensity range.

Preset level controls are also available:

| Display range | Shortcut |
|---|---|
| Darker levels | **Ctrl+D** |
| Middle levels | **Ctrl+M** |
| Brighter levels | **Ctrl+B** |

These settings change the display and, when **Current visible region** is selected during calibration, the processed region used for target detection.

## Camera geometry calibration

Camera geometry must be established before 2D points can be reconstructed in 3D. Open:

**Calibration → Camera geometry...**

The dialog currently supports:

- **ChArUco board**;
- **Checkerboard**;
- **Circle grid**, symmetric or asymmetric.

The dialog preserves its settings between uses and provides a live diagram of the selected target.

### Test before calibrating

Move to a frame in which the target is visible in both cameras and click **Test**. Testing:

- uses only the currently displayed frame pair;
- does not close the dialog;
- shows both images with detected features in yellow;
- reports the number of detected points;
- reports the number of shared corner IDs for ChArUco targets.

This is strongly recommended before running a calibration across many frames.

### Choosing calibration frames

The dialog provides four frame-selection modes:

- **Marked frames; if none, evenly spaced** — uses frames containing marker 1 in either image; if none exist, samples evenly across the video;
- **Evenly spaced frames** — samples the requested number of frames across the entire video;
- **Current frame only**;
- **Frame range** — uses a start, stop, and step.

For a good calibration, use many views of the target at different positions, distances, and orientations, including positions near the image edges.

### Detection region

- **Current visible region** searches only the region currently visible in each image view. This can exclude clutter, but the target must be inside the displayed region in both cameras.
- **Full frame** searches the entire image.

### ChArUco calibration

Specify:

- board columns and rows in **squares**;
- square size;
- marker size;
- the ArUco dictionary printed on the board;
- the legacy-pattern option when required by an older board layout.

The physical dimensions determine the units of all reconstructed coordinates. For example, dimensions entered in millimeters produce 3D coordinates in millimeters.

ChArUco calibration can use partial target views because each detected ChArUco corner has an ID. The two cameras do not need to detect exactly the same complete board in a frame; Jink3D keeps only corner IDs detected in both views before stereo calibration.

### Checkerboard calibration

Specify the number of physical board squares shown in the preview and the square size. Jink3D automatically subtracts one from each dimension to obtain the internal-corner counts expected by OpenCV.

A plain checkerboard has no individually labeled corners. Both cameras can sometimes detect every corner but return them in incompatible orders. When that happens, triangulated target points may form a line or depth-stretched pattern instead of a plane. Inspect calibration frames in the 3D view and reject frames whose reconstructed checkerboard is not planar and correctly oriented.

### Circle-grid calibration

Choose either:

- **Symmetric circle grid** — one column count and one row count;
- **Asymmetric circle grid** — alternating odd-row and even-row column counts plus the total number of rows.

For asymmetric grids, the odd and even column counts must differ by no more than one. The size field represents center-to-center spacing rather than checker size.

OpenCV circle-grid conventions vary among printed targets. Jink3D therefore tries several plausible asymmetric interpretations, with and without clustering. Use **Test** to confirm that the target is found and that the reported interpretation matches the target before running the full calibration.

### Running and inspecting calibration

Click **OK** to run detection on the selected frames and estimate:

- each camera's intrinsic matrix and lens distortion;
- rotation and translation between the cameras;
- projection matrices used for triangulation.

The console reports the stereo root-mean-square error and the frames retained for stereo calibration.

Afterward, use **Shift+Left** and **Shift+Right** to inspect detected calibration frames. Detected target points appear in yellow in the camera views and are triangulated in the 3D view when corresponding points are available in both cameras.

Save the result with:

**Calibration → Save camera geometry...**

Load an existing result with:

**Calibration → Load camera geometry...** or **Ctrl+Shift+C**.

## Orientation of gravity

Camera geometry determines how to reconstruct 3D positions. Orientation is a separate, optional step that rotates the reconstructed coordinate system so gravity points along negative z.

Camera geometry must be loaded or calculated before either orientation method.

Open:

**Calibration → Orientation of gravity...**

### Plumbline

1. Mark the top of a freely hanging line or object with marker 1.
2. Mark the bottom with marker 2.
3. Mark both endpoints in both camera views.
4. Use one or more frames.
5. Select the plumbline method and click **OK**.

Jink3D averages the reconstructed vector from marker 1 to marker 2 and rotates it to `(0, 0, -1)`.

### Thrown object

1. Mark the center of a freely thrown object with one marker in both cameras.
2. Use at least three shared marked frames; more frames are generally better.
3. Select the marker and enter the frame rate.
4. Click **OK**.

Jink3D reconstructs the 3D trajectory, fits splines to x, y, and z through time, and uses the mean second derivative as the gravity vector. The console reports the acceleration vector and its magnitude in calibration-units per second squared.

If matching `.cfg` files exist beside both videos and contain a `[Record]` section with an `fps` value, the dialog uses that frame rate. Otherwise it defaults to 30 fps and should be corrected manually.

When the calibration dimensions are in meters, the acceleration magnitude should be near 9.81 m/s². When they are in millimeters, it should be near 9810 mm/s².

Save the result with:

**Calibration → Save orientation...**

Load an existing result with:

**Calibration → Load orientation...** or **Ctrl+Shift+P**.

Without a saved orientation, Jink3D uses the camera-derived coordinate system without an additional gravity rotation.

## Saving and loading data

### Marker data

Marker data contain the manually placed and interpolated 2D coordinates for both camera views.

- **File → Load marker data...** — **Ctrl+L**
- **File → Save marker data** — **Ctrl+S**
- **File → Save marker data as...** — **Ctrl+Shift+S**

Marker data are saved as NumPy `.npy` files. Undo history is not saved.

### Camera geometry

Camera-geometry files are NumPy arrays containing camera matrices, projection matrices, distortion coefficients, and the stereo rotation and translation. Suggested filenames begin with `C`.

### Orientation

Orientation files contain the 3 × 3 gravity-alignment rotation matrix. Suggested filenames begin with `P`.

### 3D exports

- **File → Export 3D data as npy...** — **Alt+S**
- **File → Export 3D data as npy as...**
- **File → Export 3D data as CSV**
- **File → Export 3D data as CSV as...**

The native 3D array is organized as markers × coordinates × frames. The NumPy export preserves that structure. The CSV pathway should be treated as provisional because CSV is inherently two-dimensional and the current internal data are three-dimensional.

## Recommended workflow

1. Record synchronized left and right videos of a calibration target.
2. Load the calibration videos.
3. Open **Camera geometry...**, enter the target parameters, and test a good frame.
4. Run calibration across varied frames.
5. Inspect retained frames with **Shift+Left** and **Shift+Right** and verify sensible, planar target geometry.
6. Save the camera geometry.
7. Load the experimental stereo videos.
8. Load the camera geometry.
9. Optionally determine gravity orientation with a plumbline or thrown object and save it.
10. Mark corresponding points in both views.
11. Inspect interpolation at interval midpoints.
12. Save marker data and export the reconstructed 3D coordinates.

## Current limitations

- exactly two cameras;
- AVI input;
- synchronized videos with matching frame counts are expected;
- nine marker identities;
- camera geometry must remain unchanged between calibration and experimental recording;
- no wand-based camera calibration yet;
- circle-grid support should be tested carefully for each printed target;
- plain checkerboard corner ordering can occasionally be inconsistent between cameras.
