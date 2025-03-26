# Camera calibration tool

This tool helps with camera calibration

## Usage

Print chessboard and stick it on solid surface
Example chessboard pattern can be found in [OpenCV repository](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)

Then take 15+ photos of chessboard from different angles, store photos in directory

Run script:
```bash
python camera-coeffs.py --images /path/to/images --chessboard-h 9 --chessboard-w 6
```

Script will print camera coefficients

Script is adaptation of [OpenCV documentation](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
