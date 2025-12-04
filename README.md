Invisible Cloak (OpenCV) �
A real-time computer vision application that creates an "invisibility cloak" effect using color segmentation and background subtraction.

 Overview
This project simulates the invisibility cloak concept from Harry Potter. It captures the background and replaces the user's "cloaked" area (defined by a specific color range) with the pre-captured background pixels, creating a transparency effect.

Key Features:

Dynamic Color Selection: Supports Red, Blue, and Custom calibration.
Custom Calibration ('Hoodie Mode'): specifically tuned algorithm to handle low-saturation colors (like dark brown/grey) using refined HSV ranges to distinguish fabric from shadows and skin tone.
Noise Removal: Implements Morphological Operations (Opening & Dilation) to ensure a smooth mask without graininess.
🛠 Tech Stack
Language: Python 3
Libraries: OpenCV (cv2), NumPy
⚙ How it Works
Background Capture: The camera records the static background for 3 seconds.
Color Space Conversion: Frames are converted from BGR to HSV (Hue, Saturation, Value).
Masking: A specific color range is segmented out.
Bitwise Operations:
The Mask is inverted to segment the user.
The Original Background is applied to the masked area.
The two frames are combined to create the illusion.
 How to Run
Install dependencies:
pip install opencv-python numpy
