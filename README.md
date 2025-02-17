
# Setup
To set up the environment on your local machine;

Clone the repository;
`git clone https://github.com/colm-brandon-ul/segarray-code.git`

then `cd` into the repo directory using:

`cd segarray-code`


## On MacOS / Linux

Run this command in terminal, this will set up the python environment and install the dependencies:
`source setup.sh`

## On Windows
Run this command in command prompt or powershell:
`cmd /k setup.bat`

# Inference
Run the following command to make predictions:
`python main.py --img path_to_your_image_file `

This will output the source image with the bounding boxes overlayed to the `output` directory.