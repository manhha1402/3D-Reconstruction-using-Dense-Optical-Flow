# 3D-Reconstruction-using-Dense-Optical-Flow
This is an another simple version of 3D Reconstruction. I use feature dectection and matching to estimate projection matrices. In stead of
using the features that are detected from feature detection, I use Farneback - Dense Optical Flow method to detect the small change of 
every pixels in a pair of stereo images. It results in the matching of every pixels that I use to reconstruct 3D points. Again, I did not
use Bundle Adjustment to refine the result. I will try to modify the program later.
Dependencies : OpenCV, PCL
